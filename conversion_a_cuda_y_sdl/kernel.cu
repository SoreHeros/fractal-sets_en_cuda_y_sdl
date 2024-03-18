
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SDL.h>
#undef main

#include <math.h>
#include <time.h>
#include <stdio.h>

#define ZOOM 0.05
#define BLOCK_SIDE 32
#define ITER_COLOR 1024

#define cx_ini 0
#define cy_ini 0
#define upp_ini 0.002

#define PIXEL_NEGRO 0x000000ff
#define PIXEL_ROJO  0xff0000ff
#define PIXEL_AZUL  0x0000ffff
#define PIXEL_VERDE 0x00ff00ff

#define theta_k 0

#define function(Z, C) sum_c(mult_c(Z, Z), C)

//todo incluir método de newton

//cosas a ser cambiables por usuario
#define POINT_LEN 20
#define POINT_RADIUS 2
#define strength 1

#define fuerza 10

enum { LEFT_CLICK = 0b1, RIGTH_CLICK = 0b10, MIDDLE_CLICK = 0b100, MOUSE_X1 = 0b10000, MOUSE_X2 = 0b100000 } mouse_buttons;

enum { IZQUIERDA, DERECHA, ABAJO, ARRIBA};

struct {
    int posx;
    int posy;
    int deltax;
    int deltay;
    int button;
    int wheel;
}mouse = { 0 };

char keyboard[256] = {0};

typedef struct {
    double real;
    double imag;
}complex;

typedef struct {
    complex C;
    int iter;
    int shown;
}count;

typedef struct {
    complex center;
    double upp;
    SDL_Surface* surface;
} complex_area;

struct {
    int len;
    complex origen;
    count coords[POINT_LEN];
} point = {0};

struct {
    complex center;
    double upp;
}new_cords = { 0 };

typedef union{
    unsigned int full; //AZUL: 0x0000ffff
    struct { // no cambiar orden
        unsigned char alpha;
        unsigned char blue;
        unsigned char green;
        unsigned char red;
    };
}pixel;

struct {
    count* comp_cords;
    count* copy_cords;
    pixel* pixel_arr;
}gpu_mem = { 0 };

complex Z0 = { 0 };

__device__ __host__ complex mult_c(complex A, complex B) {
    complex R;

    R.real = A.real * B.real - A.imag * B.imag;
    R.imag = A.real * B.imag + A.imag * B.real;

    return R;
}

__device__ __host__ complex sum_c(complex A, complex B) {
    complex R;

    R.real = A.real + B.real;
    R.imag = A.imag + B.imag;

    return R;
}

__device__ __host__ double sqrabs_c(complex C) {
    return C.real * C.real + C.imag * C.imag;
}

__device__ __host__ double abs_c(complex C) {
    return sqrt(sqrabs_c(C));
}

__device__ __host__ double theta_c(complex C) {
    double r = (C.real == 0 ? M_PI * 0.5 : atan(C.imag / C.real));

    if (C.real <= 0) {
        r += M_PI;
    }
    
    if (C.imag < 0 && C.real < 0) {
        r -= 2 * M_PI;
    }

    r += 2 * M_PI * theta_k;

    return r;
}

__device__ __host__ complex log_c(complex C) {
    return { log(abs_c(C)), theta_c(C) };
}

__device__ __host__ complex pow_c(complex A, complex B) {
    if (A.real == 0 && A.imag == 0)
        return{ 0, 0 };

    complex p = mult_c(B, log_c(A));

    complex ER = { exp(p.real),0 };

    complex EI = { cos(p.imag), sin(p.imag) };

    complex R = mult_c(ER, EI);

    return R;
}

//devuelve un color basandose en la cuenta y el cutoff que le digas
__device__ __host__ pixel pixel_color(int count) {
    #define local 0.5

    unsigned char red, blue, green;

    double u = count / (double)ITER_COLOR;

    u -= (int)u;

    pixel pix;

    //u = pow(u, strength);

    //gran código para la gpu 😎

    //SECCIÓN:           CRECER DESDE UN PUNTO A OTRO                    PERMANECER A MAX BRILLO                        DECRECER EN BRILLO

    pix.blue = (char)(0xff * (u - 0.8) * 5 * (0.8 <= u && u < 1)) | (0xff * (u < 0.2)) | (char)(0xff * (0.4 - u) * 5 * (0.2 <= u && u < 0.4));

    pix.green = (char)(0xff * (u) * 5 * (u < 0.2)) | (0xff * (0.2 <= u && u < 0.6)) | (char)(0xff * (0.8 - u) * 5 * (0.6 <= u && u < 0.8));

    pix.red = (char)(0xff * (u - 0.4) * 5 * (0.4 <= u && u < 0.6)) | (0xff * (0.6 <= u && u < 0.8)) | (char)(0xff * (1 - u) * 5 * (0.8 <= u && u < 1));;

    pix.alpha = 0xff;

    return pix;
}

//inicializa los pixels para empezar las iteraciones
__global__ void starting_iter_funct(count* arr, int sizx, int sizy, complex center, double units_per_pixel, complex Z0, int state, int B, int T, int L, int R) {

    int x = (blockIdx.x * BLOCK_SIDE) % sizx + threadIdx.x % BLOCK_SIDE;
    int y = (blockIdx.x * BLOCK_SIDE) / sizx * BLOCK_SIDE + threadIdx.x / BLOCK_SIDE;

    if (x < 0 || x < L || x >= sizx || x >= R || y < 0 || y < B || y >= sizy || y >= T)//comprobar si está dentro del array
        return;
    

    //se divide en sizx * sizy / BLOCK_SIZE + 1 bloques, los cuales cubren un área de 32 x 32 píxeles cada uno
    complex Z = { state * (center.real + units_per_pixel * (x - (sizx / 2))) + !state * Z0.real, state * (center.imag + units_per_pixel * -(y - (sizy / 2))) + !state * Z0.imag };

    /* codigo anterior unwrapped
    if (state) {
        Z.real = center.real + units_per_pixel * (x - (sizx / 2));
        Z.imag = center.imag + units_per_pixel * -(y - (sizy / 2));
    }
    else {
        Z = Z0;
    }
    */

    //aplicar color
    arr[y * sizx + x].C = Z;
    arr[y * sizx + x].iter = 0;
    arr[y * sizx + x].shown = 0;
}

//todo posible mejora, asignar la dirección correcta al principio, en vez de buscarla cada véz
__global__ void iter_calculation_funct(count* arr, int sizx, int sizy, complex center, double units_per_pixel, complex Z0, int state, int B, int T, int L, int R) {

    int x = (blockIdx.x * BLOCK_SIDE) % sizx + threadIdx.x % BLOCK_SIDE;
    int y = (blockIdx.x * BLOCK_SIDE) / sizx * BLOCK_SIDE + threadIdx.x / BLOCK_SIDE;


    if (x < 0 || x < L || x >= sizx || x >= R || y < 0 || y < B || y >= sizy || y >= T)//comprobar si está dentro del array
        return;

    //se divide en sizx * sizy / BLOCK_SIZE + 1 bloques, los cuales cubren un área de 32 x 32 píxeles cada uno

    complex C = { !state * (center.real + units_per_pixel * (x - (sizx / 2))) + state * Z0.real, !state * (center.imag + units_per_pixel * -(y - (sizy / 2))) + state * Z0.imag };
    complex Z = { state * (center.real + units_per_pixel * (x - (sizx / 2))) + !state * Z0.real, state * (center.imag + units_per_pixel * -(y - (sizy / 2))) + !state * Z0.imag };

    /* código anterior unwrapped
    if (state) {
        C = Z0;

        Z.real = center.real + units_per_pixel * (x - (sizx / 2));
        Z.imag = center.imag + units_per_pixel * -(y - (sizy / 2));
    }
    else {
        C.real = center.real + units_per_pixel * (x - (sizx / 2));
        C.imag = center.imag + units_per_pixel * -(y - (sizy / 2));

        Z = Z0;
    }
    */

    arr[y * sizx + x].C = function(arr[y * sizx + x].C, C);
    arr[y * sizx + x].iter++;

    double len = sqrabs_c(arr[y * sizx + x].C);

    if (len < INFINITY) {
        if (arr[y * sizx + x].iter > arr[y * sizx + x].shown)
            arr[y * sizx + x].shown = 0;
    }else{
        //poner la iter y reiniciar el proceso
        arr[y * sizx + x].shown = arr[y * sizx + x].iter;
        arr[y * sizx + x].iter = 0;
        arr[y * sizx + x].C = Z;
    }
}

//transformar el array a la nueva posición
__global__ void gpu_arr_transform(count * old_arr, count * new_arr,complex center, double upp, complex old_center, double oupp, int sizx, int sizy) {
    int newx = (blockIdx.x * BLOCK_SIDE) % sizx + threadIdx.x % BLOCK_SIDE;                 //get posición
    int newy = (blockIdx.x * BLOCK_SIDE) / sizx * BLOCK_SIDE + threadIdx.x / BLOCK_SIDE;

    if (newx < 0 || newx >= sizx || newy < 0 || newy >= sizy)//comprobar si está dentro del array
        return;

    double tempx = (newx - sizx / 2) * (upp / oupp) + sizx / 2; // reducir o aumentar el array
    double tempy = (newy - sizy / 2) * (upp / oupp) + sizy / 2;

    tempx -= (old_center.real - center.real) / oupp;  //moverlo a la posición adecuada
    tempy += (old_center.imag - center.imag) / oupp;

    int oldx = tempx;
    int oldy = tempy;

    tempx -= oldx;
    tempy -= oldy;
    
    if (oldx >= 1 && oldx < sizx - 1 && oldy >= 1 && oldy < sizy - 1 && newx >= 1 && newx < sizx - 1 && newy >= 1 && newy < sizy - 1) {
        new_arr[newy * sizx + newx].shown = (old_arr[oldy * sizx + oldx].shown * (1 - tempx) + old_arr[oldy * sizx + oldx + 1].shown * (tempx)) * (1 - tempy) + (old_arr[(oldy + 1) * sizx + oldx].shown * (1 - tempx) + old_arr[(oldy + 1) * sizx + oldx + 1].shown * (tempx)) * (tempy);
        if (sqrabs_c(old_arr[oldy * sizx + oldx].C) < INFINITY) {
            new_arr[newy * sizx + newx].C = old_arr[oldy * sizx + oldx].C;
        }
    }
}

//todo con la normal multiplicar el color del pixel por el producto escalar de la normal y la dirección de la lúz
__device__ pixel temporal_funct(pixel pix, int ligth) {

    if (pix.red + ligth > 0xff)
        pix.red = 0xff;
    else if (pix.red + ligth < 0x00)
        pix.red = 0;
    else
        pix.red += ligth;


    if (pix.blue + ligth > 0xff)
        pix.blue = 0xff;
    else if (pix.blue + ligth < 0x00)
        pix.blue = 0;
    else
        pix.blue += ligth;


    if (pix.green + ligth > 0xff)
        pix.green = 0xff;
    else if (pix.green + ligth < 0x00)
        pix.green = 0;
    else
        pix.green += ligth;


    return pix;
}

__global__ void get_pixel_funct(count* arr, pixel* pixels, int sizx, int sizy) {

    int x = (blockIdx.x * BLOCK_SIDE) % sizx + threadIdx.x % BLOCK_SIDE;
    int y = (blockIdx.x * BLOCK_SIDE) / sizx * BLOCK_SIDE + threadIdx.x / BLOCK_SIDE;


    if (x < 0 || x >= sizx - 1 || y < 1 || y >= sizy)//comprobar si está dentro del array
        return;

    //convertir a pixel
    if (arr[y * sizx + x].shown) {
        pixels[y * sizx + x] = pixel_color(arr[y * sizx + x].shown);

        int ligth = (arr[(y - 1) * sizx + x].shown + arr[y * sizx + x + 1].shown) / 2 - arr[y * sizx + x].shown;

        //pixels[y * sizx + x] = temporal_funct(pixels[y * sizx + x], );
        //todo: sacar normal y utilizarla para averiguar la lúz
    }else
        pixels[y * sizx + x].full = PIXEL_NEGRO;
}


//todo: carga en rectáncgulos al moverse y alejarse, carga especial para acercarse, usar interpolación?
cudaError_t rellenar_pixels(complex_area* area, complex Z0, int state) {

    cudaError_t error;
    
    //mover el array si es necesario (y recalcular cosas)
    if (area->center.real != new_cords.center.real || area->center.imag != new_cords.center.imag || area->upp != new_cords.upp) {
        //inicializar nuevo array
        starting_iter_funct <<< (area->surface->w / (BLOCK_SIDE)+1) * (area->surface->h / (BLOCK_SIDE)+1), BLOCK_SIDE* BLOCK_SIDE >>> (gpu_mem.copy_cords, area->surface->w, area->surface->h, area->center, area->upp, Z0, state, 0, area->surface->h, 0, area->surface->w);

        //cudaDeviceSynchronize();

        //copiar la muestra del array anterior
        gpu_arr_transform <<<(area->surface->w / BLOCK_SIDE + 1) * (area->surface->h / BLOCK_SIDE + 1), BLOCK_SIDE* BLOCK_SIDE >>> (gpu_mem.comp_cords, gpu_mem.copy_cords, new_cords.center, new_cords.upp,area->center, area->upp, area->surface->w, area->surface->h);

        count* aux = gpu_mem.comp_cords;
        gpu_mem.comp_cords = gpu_mem.copy_cords;
        gpu_mem.copy_cords = aux;
        
        area->center = new_cords.center;
        area->upp = new_cords.upp;

        //cudaDeviceSynchronize();
    }

    //calcular próxima iteración
    iter_calculation_funct <<< (area->surface->w / (BLOCK_SIDE) + 1) * (area->surface->h / (BLOCK_SIDE) + 1), BLOCK_SIDE * BLOCK_SIDE >>> (gpu_mem.comp_cords, area->surface->w, area->surface->h, area->center, area->upp, Z0, state, 0, area->surface->h, 0, area->surface->w);

    //cudaDeviceSynchronize();

    //colorear pixels
    get_pixel_funct <<< (area->surface->w / (BLOCK_SIDE)+1) * (area->surface->h / (BLOCK_SIDE)+1), BLOCK_SIDE* BLOCK_SIDE >>> (gpu_mem.comp_cords, gpu_mem.pixel_arr, area->surface->w, area->surface->h);

    //cudaDeviceSynchronize();

    //copiar mem
    error = cudaMemcpy(area->surface->pixels, gpu_mem.pixel_arr, area->surface->w * area->surface->h * sizeof(int), cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto ERROR;
    }

ERROR:
    return error;
}

void draw_red_point(complex coords, pixel* arr, int sizx, int sizy, complex center, double units_per_pixel) {

    int x =  (coords.real - center.real) / units_per_pixel;
    int y = -(coords.imag - center.imag) / units_per_pixel;

    x += sizx / 2;
    y += sizy / 2;

    if (x >= 0 && x < sizx && y >= 0 && y < sizy) {
        for (int j = -POINT_RADIUS, i; j <= POINT_RADIUS; j++)
            for (i = -POINT_RADIUS; i <= POINT_RADIUS; i++)
                if (x + i >= 0 && x + i < sizx && y + j >= 0 && y + j < sizy)
                    arr[(y + j) * sizx + x + i].full = PIXEL_ROJO;
    }
}

void draw_point(count coords, pixel* arr, int sizx, int sizy, complex center, double units_per_pixel) {

    int x =  (coords.C.real - center.real) / units_per_pixel;
    int y = -(coords.C.imag - center.imag) / units_per_pixel;

    x += sizx / 2;
    y += sizy / 2;

    if (x >= 0 && x < sizx && y >= 0 && y < sizy) {
        for(int j = -POINT_RADIUS, i; j <= POINT_RADIUS; j++)
            for(i = - POINT_RADIUS; i <= POINT_RADIUS; i++)
                if (x + i >= 0 && x + i < sizx && y + j >= 0 && y + j < sizy) {
                    if ((i == -POINT_RADIUS || i == POINT_RADIUS) && (j == -POINT_RADIUS || j == POINT_RADIUS))
                        arr[(y + j) * sizx + x + i].full = PIXEL_AZUL | PIXEL_ROJO | PIXEL_VERDE; // rellenar color esquinas
                    else
                        arr[(y + j) * sizx + x + i] = pixel_color(coords.iter); // rellenar color normal
                }
    }
}

void draw_line(complex p1, complex p2, pixel* arr, int sizx, int sizy, complex center, double units_per_pixel) {

    long long int x1 =  (p1.real - center.real) / units_per_pixel;
    long long int y1 = -(p1.imag - center.imag) / units_per_pixel;
    long long int x2 =  (p2.real - center.real) / units_per_pixel;
    long long int y2 = -(p2.imag - center.imag) / units_per_pixel;

    x1 += sizx / 2;
    y1 += sizy / 2;
    x2 += sizx / 2;
    y2 += sizy / 2;

    double t;

    long long int x = x1 - x2;
    long long int y = y1 - y2;

    if (x < 0)
        x = -x;

    if (y < 0)
        y = -y;

    long long int divisions = (x + y);

    long long int i = 0;

    if (x1 < 0 && x1 != x2)
        if (i < (0    - x1) * divisions / (x2 - x1)) // está fuera por la izquierda
            i = (0    - x1) * divisions / (x2 - x1) + 1;

    if (x1 >= sizx && x1 != x2)
        if (i < (sizx - x1) * divisions / (x2 - x1)) // está fuera por la derecha
            i = (sizx - x1) * divisions / (x2 - x1) + 1;

    if (y < 0 && y1 != y2)
        if (i < (0    - y1) * divisions / (y2 - y1)) // está fuera por arriba
            i = (0    - y1) * divisions / (y2 - y1) + 1;

    if (y >= sizy && y1 != y2)
        if (i < (sizy - y1) * divisions / (y2 - y1)) // está fuera por abajo
            i = (sizy - y1) * divisions / (y2 - y1) + 1;

    for (; i <= divisions; i++) {
        t = i / (double)divisions;
        x = x1 * (1 - t) + x2 * t;
        y = y1 * (1 - t) + y2 * t;
        if (x < 0 || x >= sizx || y < 0 || y >= sizy)
            break;

        arr[y * sizx + x].full = PIXEL_AZUL | PIXEL_ROJO | PIXEL_VERDE;
    }

    //todo: fix
}

void show_points(pixel *arr, int sizx, int sizy, complex center, double units_per_pixel) {
    if (point.len) {
        for (int i = 1; i < POINT_LEN; i++)
            draw_line(point.coords[i - 1].C, point.coords[i].C, arr, sizx, sizy, center, units_per_pixel);
        for(int i = 0; i < POINT_LEN; i++)
            draw_point(point.coords[i], arr, sizx, sizy, center, units_per_pixel);
        draw_red_point(point.origen, arr, sizx, sizy, center, units_per_pixel);
        draw_red_point(Z0, arr, sizx, sizy, center, units_per_pixel);
    }
}

void calculate_points(int state) {
    complex C, Z;

    if (state) {
        C = Z0;
        Z = point.origen;
    }
    else {
        C = point.origen;
        Z = Z0;
    }

    if (point.len == 0 || sqrabs_c(point.coords[0].C) >= INFINITY)
        for (int i = 0; i < POINT_LEN; i++) {
            point.coords[i].C = Z = function(Z, C);
            point.coords[i].iter = i + 1;
        }
    else
        for (int i = 0; i < POINT_LEN; i++) {
            point.coords[i].C = function(point.coords[i].C, C);
            point.coords[i].iter++;
        }

    point.len = 1;
}


int main(int argc, char* argv[]){

    SDL_Init(SDL_INIT_EVERYTHING);

    SDL_DisplayMode monitor_display;

    SDL_GetDesktopDisplayMode(0, &monitor_display);

    SDL_Window* window = SDL_CreateWindow("Fractals", 0, 0, monitor_display.w, monitor_display.h, SDL_WINDOW_BORDERLESS);

    SDL_SetWindowDisplayMode(window, &monitor_display);

    SDL_Surface* screen = SDL_GetWindowSurface(window);

    complex_area calculated_area;
    calculated_area.surface = SDL_CreateRGBSurfaceWithFormat(0, monitor_display.w, monitor_display.h, 32, SDL_PIXELFORMAT_RGBX8888);

    printf("ancho: %i\naltura: %i\n", monitor_display.w, monitor_display.h);

    cudaError_t cudaStatus = cudaSetDevice(0);
    
    calculated_area.center.real = cx_ini;
    calculated_area.center.imag = cy_ini;
    calculated_area.upp = upp_ini;

    new_cords.center = calculated_area.center;
    new_cords.upp = calculated_area.upp;

    int state = 0, s_case = 1, force_new_arr = 0;

    cudaStatus = cudaMalloc(&gpu_mem.pixel_arr, monitor_display.w * monitor_display.h * sizeof(int));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto END;
    }

    cudaStatus = cudaMalloc(&gpu_mem.comp_cords, monitor_display.w * monitor_display.h * sizeof(count));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto END;
    }

    cudaStatus = cudaMalloc(&gpu_mem.copy_cords, monitor_display.w * monitor_display.h * sizeof(count));

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto END;
    }

    starting_iter_funct <<<(calculated_area.surface->w / BLOCK_SIDE + 1) * (calculated_area.surface->h / BLOCK_SIDE + 1), BLOCK_SIDE* BLOCK_SIDE >>> (gpu_mem.comp_cords, calculated_area.surface->w, calculated_area.surface->h, calculated_area.center, calculated_area.upp, Z0, state, 0, calculated_area.surface->h, 0, calculated_area.surface->w);


    while (1) {
        SDL_Event ev;
        mouse.wheel = 0;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT:
                goto END;


            case SDL_KEYDOWN:
                switch (ev.key.keysym.sym) {
                case SDLK_ESCAPE:
                    goto END;

                case SDLK_MINUS:
                    keyboard['-'] = 1;
                    break;

                case SDLK_PLUS:
                    keyboard['+'] = 1;
                    break;

                case SDLK_r:
                    keyboard['r'] = 1;
                    break;

                case SDLK_s:
                    keyboard['s'] = 1;
                    break;

                case SDLK_f:
                    keyboard['f'] = 1;
                    break;

                case SDLK_p:
                    keyboard['p'] = 1;
                    break;

                }
                break;


            case SDL_KEYUP:
                switch (ev.key.keysym.sym) {
                case SDLK_MINUS:
                    keyboard['-'] = 0;
                    break;

                case SDLK_PLUS:
                    keyboard['+'] = 0;
                    break;
                    
                case SDLK_r:
                    keyboard['r'] = 0;
                    break;

                case SDLK_s:
                    keyboard['s'] = 0;
                    break;

                case SDLK_f:
                    keyboard['f'] = 0;
                    break;

                case SDLK_p:
                    keyboard['p'] = 0;
                    break;

                }
                break;


            case SDL_MOUSEBUTTONDOWN:
                switch (ev.button.button) {
                case SDL_BUTTON_RIGHT:
                    mouse.button |= RIGTH_CLICK;
                    break;

                case SDL_BUTTON_LEFT:
                    mouse.button |= LEFT_CLICK;
                    break;

                case SDL_BUTTON_MIDDLE:
                    mouse.button |= MIDDLE_CLICK;
                    break;

                case SDL_BUTTON_X1:
                    mouse.button |= MOUSE_X1;
                    break;

                case SDL_BUTTON_X2:
                    mouse.button |= MOUSE_X2;
                    break;

                }
                break;


            case SDL_MOUSEBUTTONUP:
                switch (ev.button.button) {
                case SDL_BUTTON_RIGHT:
                    mouse.button &= ~RIGTH_CLICK;
                    break;

                case SDL_BUTTON_LEFT:
                    mouse.button &= ~LEFT_CLICK;
                    break;

                case SDL_BUTTON_MIDDLE:
                    mouse.button &= ~MIDDLE_CLICK;
                    break;

                case SDL_BUTTON_X1:
                    mouse.button &= ~MOUSE_X1;
                    break;

                case SDL_BUTTON_X2:
                    mouse.button &= ~MOUSE_X2;
                    break;
                }
                break;


            case SDL_MOUSEWHEEL:
                mouse.wheel = ev.wheel.y;
            }
        }

        //get mouse for this frame
        {
            int x, y;

            SDL_GetMouseState(&x, &y);

            mouse.deltax = x - mouse.posx;
            mouse.deltay = y - mouse.posy;

            mouse.posx = x;
            mouse.posy = y;
        }

        //handle zoom
        if (mouse.wheel + keyboard['+'] - keyboard['-']) {
            double local_zoom = (mouse.wheel + keyboard['+'] - keyboard['-']);

            //zoom general
            if (local_zoom > 0)
                for (int i = 0; i <= local_zoom; i++) {
                    new_cords.center.real += (mouse.posx - monitor_display.w / 2) * new_cords.upp * ZOOM;
                    new_cords.center.imag -= (mouse.posy - monitor_display.h / 2) * new_cords.upp * ZOOM;
                    new_cords.upp *= 1 - ZOOM;
                }
            if (local_zoom < 0)
                for (int i = 0; i >= local_zoom; i--) {
                    new_cords.center.real -= (mouse.posx - monitor_display.w / 2) * new_cords.upp * ZOOM;
                    new_cords.center.imag += (mouse.posy - monitor_display.h / 2) * new_cords.upp * ZOOM;
                    new_cords.upp *= 1 + ZOOM;
                }
        }

        
        //handle mouse movement
        if (mouse.button & (LEFT_CLICK)) {
            new_cords.center.real -= mouse.deltax * new_cords.upp;
            new_cords.center.imag += mouse.deltay * new_cords.upp;//cambio de signo necesario para mantener la dirección (matemáticamente)
        }

        
        if (mouse.button & MIDDLE_CLICK) {
            int x = mouse.posx - monitor_display.w / 2;
            int y = -(mouse.posy - monitor_display.h / 2);

            Z0.real = x * new_cords.upp + new_cords.center.real;
            Z0.imag = y * new_cords.upp + new_cords.center.imag;

            force_new_arr = 1;
        }

        
        //put point
        if (mouse.button & (RIGTH_CLICK)) {
            //get primer pounto
            int x =   mouse.posx - monitor_display.w / 2 ;
            int y = -(mouse.posy - monitor_display.h / 2);

            if (point.origen.real != x * new_cords.upp + new_cords.center.real || point.origen.imag != y * new_cords.upp + new_cords.center.imag) {
                point.origen.real = x * new_cords.upp + new_cords.center.real;
                point.origen.imag = y * new_cords.upp + new_cords.center.imag;
                point.len = 0;
            }
        }

        if (keyboard['p'])
            point.len = 0;
        

        if (keyboard['r']) {
            calculated_area.center.real = cx_ini;
            calculated_area.center.imag = cy_ini;
            calculated_area.upp = upp_ini;
            Z0 = { 0 };

            new_cords.center = calculated_area.center;
            new_cords.upp = calculated_area.upp;

            force_new_arr = 1;
        }

        if (keyboard['s'] && s_case) {
            state = !state;
            s_case = 0;
            force_new_arr = 1;
        }
        
        if (!keyboard['s'])
            s_case = 1;

        if (keyboard['f'] || force_new_arr) {
            calculated_area.center = new_cords.center;
            calculated_area.upp = new_cords.upp;

            starting_iter_funct<<<(calculated_area.surface->w / BLOCK_SIDE + 1) * (calculated_area.surface->h / BLOCK_SIDE + 1), BLOCK_SIDE* BLOCK_SIDE >>>(gpu_mem.comp_cords, calculated_area.surface->w, calculated_area.surface->h, calculated_area.center, calculated_area.upp, Z0, state, 0, calculated_area.surface->h, 0, calculated_area.surface->w);
            force_new_arr = 0;
        }

        // write the pixels
        SDL_LockSurface(calculated_area.surface);

        cudaStatus = rellenar_pixels(&calculated_area, Z0, state);


        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "rellenar pixeles failed!");
            goto END;
        }

        calculate_points(state);

        show_points((pixel*)calculated_area.surface->pixels, calculated_area.surface->w, calculated_area.surface->h, calculated_area.center, calculated_area.upp);
           
        SDL_UnlockSurface(calculated_area.surface);
        
        // copy to window
        SDL_BlitSurface(calculated_area.surface, NULL, screen, NULL);
        SDL_UpdateWindowSurface(window);
    }

END:

    printf("\n");

    cudaDeviceSynchronize();
    
    cudaFree(gpu_mem.pixel_arr);
    cudaFree(gpu_mem.copy_cords);
    cudaFree(gpu_mem.comp_cords);

    SDL_Quit();

    SDL_FreeSurface(calculated_area.surface);
    SDL_DestroyWindow(window);

    return 0;
}