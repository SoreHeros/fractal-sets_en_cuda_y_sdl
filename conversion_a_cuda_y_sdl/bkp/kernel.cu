
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <SDL.h>
#undef main

#include <math.h>
#include <time.h>
#include <stdio.h>

#define ZOOM 0.05
//#define BLOCK_SIZE 1024
#define BLOCK_SIDE 32
#define SQUARED_RADIUS INFINITY

#define cx_ini 0
#define cy_ini 0
#define upp_ini 0.002

#define temp_real 2
#define temp_imag 0

#define theta_k 0

#define temp {temp_real, temp_imag}

//cosas a ser cambiables por usuario
#define SAMPLE_DEPTH 1024
#define POINT_RADIUS 2
#define strength 1

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
    complex center;
    double upp;
    SDL_Surface* surface;
} complex_area;

struct {
    int len;
    complex* coords;
} point = {0};

struct {
    int len;
    int B[10];
    int T[10];
    int L[10];
    int R[10];
}rect = { 0 };

struct {
    int* main_arr;
    int* copy_arr;
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

//todo fix power para nums complejos
__device__ __host__ complex pow_c(complex A, complex B) {
    if (A.real == 0 && A.imag == 0)
        return{ 0, 0 };

    complex p = mult_c(B, log_c(A));

    complex ER = { exp(p.real),0 };

    complex EI = { cos(p.imag), sin(p.imag) };

    complex R = mult_c(ER, EI);

    return R;
}

// funcion de cargado original ~1100ms per frame
/*
__global__ void gpu_pixel_funct(int * arr, int sizx, int sizy, double cx, double cy, double units_per_pixel, complex Z0, int state){

    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < sizx * sizy) {//comprobar si está dentro del array



        int x = i % sizx;
        int y = i / sizx;

        complex C, Z;

        
        if (state) {
            Z.real = cx + units_per_pixel * (x - (sizx / 2));
            Z.imag = cy + units_per_pixel * -(y - (sizy / 2));

            C = Z0;
        }
        else {
            C.real = cx + units_per_pixel * (x - (sizx / 2));
            C.imag = cy + units_per_pixel * -(y - (sizy / 2));

            Z = Z0;
        }
        

        int k;


        for (k = 0; k < SAMPLE_DEPTH && sqrabs(Z) <= SQUARED_RADIUS; k++) {
            Z = sum(pow_c(Z, temp), C);
        }

        double u;

        unsigned char red, blue, green;

        u = k / (double)SAMPLE_DEPTH;

        //u = pow(u, strength);

        //gran código para la gpu 😎

        //SECCIÓN:           CRECER DESDE UN PUNTO A OTRO                    PERMANECER A MAX BRILLO                        DECRECER EN BRILLO
        
        blue  =                                                         (0xff * (             u < 0.25)) | (char)(0xff * (0.5 - u) * 4 * (0.25 <= u && u < 0.5));

        green = (char)(0xff * (u      ) * 4 * (            u < 0.25)) | (0xff * (0.25 <= u && u < 0.75)) | (char)(0xff * (1   - u) * 4 * (0.75 <= u && u < 1  ));

        red   = (char)(0xff * (u - 0.5) * 4 * (0.5 <= u && u < 0.75)) | (0xff * (0.75 <= u && u < 1   ))                                                        ;

        //aplicar color
        arr[y * sizx + x] = (red << 24) | (green << 16) | (blue << 8) | 0x000000ff;
    }
}
*/

//nueva carga de datos ~900 ms

__global__ void force_pixel_funct(int* arr, int sizx, int sizy, complex center, double units_per_pixel, complex Z0, int state, int B, int T, int L, int R) {

    int blockx = (blockIdx.x * BLOCK_SIDE) % sizx;
    int blocky = (blockIdx.x * BLOCK_SIDE) / sizx * BLOCK_SIDE;


    int x = blockx + threadIdx.x % BLOCK_SIDE;
    int y = blocky + threadIdx.x / BLOCK_SIDE;


    if (x >= sizx || y >= sizy || x >= R || y >= T)//comprobar si está dentro del array
        return;
    
    //se divide en sizx * sizy / BLOCK_SIZE + 1 bloques, los cuales cubren un área de 32 x 32 píxeles cada uno

    complex C, Z;

    if (state) {
        Z.real = center.real + units_per_pixel * (x - (sizx / 2));
        Z.imag = center.imag + units_per_pixel * -(y - (sizy / 2));

        C = Z0;
    }
    else {
        C.real = center.real + units_per_pixel * (x - (sizx / 2));
        C.imag = center.imag + units_per_pixel * -(y - (sizy / 2));

        Z = Z0;
    }


    int k;

    for (k = 0; k < SAMPLE_DEPTH && sqrabs_c(Z) <= SQUARED_RADIUS; k++) {
        Z = sum_c(pow_c(Z, temp), C);
    }

    double u;

    unsigned char red, blue, green;

    u = k / (double)SAMPLE_DEPTH;

    //u = pow(u, strength);

    //gran código para la gpu 😎

    //SECCIÓN:           CRECER DESDE UN PUNTO A OTRO                    PERMANECER A MAX BRILLO                        DECRECER EN BRILLO

    blue = (0xff * (u < 0.25)) | (char)(0xff * (0.5 - u) * 4 * (0.25 <= u && u < 0.5));

    green = (char)(0xff * (u) * 4 * (u < 0.25)) | (0xff * (0.25 <= u && u < 0.75)) | (char)(0xff * (1 - u) * 4 * (0.75 <= u && u < 1));

    red = (char)(0xff * (u - 0.5) * 4 * (0.5 <= u && u < 0.75)) | (0xff * (0.75 <= u && u < 1));

    //aplicar color
    arr[y * sizx + x] = (red << 24) | (green << 16) | (blue << 8) | 0x000000ff;
}

//nueva carga de datos ~450 ms (necesita arreglarse problemas) es peor????

__global__ void gpu_pixel_funct(int* arr, int sizx, int sizy, complex center, double units_per_pixel, complex Z0, int state, int resolution, int B, int T, int L, int R) {

    int blockx = (blockIdx.x * BLOCK_SIDE) % sizx;
    int blocky = (blockIdx.x * BLOCK_SIDE) / sizx * BLOCK_SIDE;


    int x = blockx + threadIdx.x % BLOCK_SIDE;
    int y = blocky + threadIdx.x / BLOCK_SIDE;


    if (x >= sizx || y >= sizy || x >= R || y >= T)//comprobar si está dentro del array
        return;

    if (blockx + BLOCK_SIDE < sizx && arr[blocky * sizx + blockx] == arr[blocky * sizx + blockx + BLOCK_SIDE] && blocky + BLOCK_SIDE < sizy && arr[blocky * sizx + blockx] == arr[(blocky + 32) * sizx + blockx] && arr[blocky * sizx + blockx] == arr[(blocky + BLOCK_SIDE) * sizx + blockx + BLOCK_SIDE]) {
        arr[y * sizx + x] = arr[blocky * sizx + blockx];
    }
    else {
        //se divide en sizx * sizy / BLOCK_SIZE + 1 bloques, los cuales cubren un área de 32 x 32 píxeles cada uno

        complex C, Z;

        if (state) {
            Z.real = center.real + units_per_pixel * (x - (sizx / 2));
            Z.imag = center.imag + units_per_pixel * -(y - (sizy / 2));

            C = Z0;
        }
        else {
            C.real = center.real + units_per_pixel * (x - (sizx / 2));
            C.imag = center.imag + units_per_pixel * -(y - (sizy / 2));

            Z = Z0;
        }


        int k;

        for (k = 0; k < SAMPLE_DEPTH && sqrabs_c(Z) <= SQUARED_RADIUS; k++) {
            Z = sum_c(pow_c(Z, temp), C);
        }

        double u;

        unsigned char red, blue, green;

        u = k / (double)SAMPLE_DEPTH;

        //u = pow(u, strength);

        //gran código para la gpu 😎

        //SECCIÓN:           CRECER DESDE UN PUNTO A OTRO                    PERMANECER A MAX BRILLO                        DECRECER EN BRILLO

        blue = (0xff * (u < 0.25)) | (char)(0xff * (0.5 - u) * 4 * (0.25 <= u && u < 0.5));

        green = (char)(0xff * (u) * 4 * (u < 0.25)) | (0xff * (0.25 <= u && u < 0.75)) | (char)(0xff * (1 - u) * 4 * (0.75 <= u && u < 1));

        red = (char)(0xff * (u - 0.5) * 4 * (0.5 <= u && u < 0.75)) | (0xff * (0.75 <= u && u < 1));

        //aplicar color
        arr[y * sizx + x] = (red << 24) | (green << 16) | (blue << 8) | 0x000000ff;
    }
}

__global__ void pre_gpu_pixel_funct(int* arr, int sizx, int sizy, complex center, double units_per_pixel, complex Z0, int state, int B, int T, int L, int R) {

    int x = (blockIdx.x * BLOCK_SIDE * BLOCK_SIDE) % sizx + (threadIdx.x % BLOCK_SIDE) * BLOCK_SIDE;
    int y = (blockIdx.x * BLOCK_SIDE * BLOCK_SIDE) / sizx * BLOCK_SIDE + (threadIdx.x / BLOCK_SIDE) * BLOCK_SIDE;

    if (x >= sizx || y >= sizy || x >= R || y >= T)//comprobar si está dentro del array
        return;

    complex C, Z;


    if (state) {
        Z.real = center.real + units_per_pixel * (x - (sizx / 2));
        Z.imag = center.imag + units_per_pixel * -(y - (sizy / 2));

        C = Z0;
    }
    else {
        C.real = center.real + units_per_pixel * (x - (sizx / 2));
        C.imag = center.imag + units_per_pixel * -(y - (sizy / 2));

        Z = Z0;
    }


    int k;

    for (k = 0; k < SAMPLE_DEPTH && sqrabs_c(Z) <= SQUARED_RADIUS; k++) {
        Z = sum_c(pow_c(Z, temp), C);
    }

    double u;

    unsigned char red, blue, green;

    u = k / (double)SAMPLE_DEPTH;

    //u = pow(u, strength);

    //gran código para la gpu 😎

    //SECCIÓN:           CRECER DESDE UN PUNTO A OTRO                    PERMANECER A MAX BRILLO                        DECRECER EN BRILLO

    blue = (0xff * (u < 0.25)) | (char)(0xff * (0.5 - u) * 4 * (0.25 <= u && u < 0.5));

    green = (char)(0xff * (u) * 4 * (u < 0.25)) | (0xff * (0.25 <= u && u < 0.75)) | (char)(0xff * (1 - u) * 4 * (0.75 <= u && u < 1));

    red = (char)(0xff * (u - 0.5) * 4 * (0.5 <= u && u < 0.75)) | (0xff * (0.75 <= u && u < 1));

    //aplicar color
    arr[y * sizx + x] = (red << 24) | (green << 16) | (blue << 8) | 0x000000ff;
}

//copiar array ~10 ms
__global__ void gpu_arr_transform(int * old_arr, int * new_arr,complex center, double upp, complex old_center, double oupp, int sizx, int sizy) {
    int newx = (blockIdx.x * BLOCK_SIDE) % sizx + threadIdx.x % BLOCK_SIDE;                 //get posición
    int newy = (blockIdx.x * BLOCK_SIDE) / sizx * BLOCK_SIDE + threadIdx.x / BLOCK_SIDE;


    double tempx = (newx - sizx / 2) * (oupp / upp) + sizx / 2; // reducir o aumentar el array
    double tempy = (newy - sizy / 2) * (oupp / upp) + sizy / 2;

    tempx -= (center.real - old_center.real) / upp;  //moverlo a la posición adecuada
    tempy += (center.imag - old_center.imag) / upp;

    int oldx = tempx;
    int oldy = tempy;

    if (oldx >= 0 && oldx < sizx && oldy >= 0 && oldy < sizy && newx >= 0 && newx < sizx && newy >= 0 && newy < sizy)
        new_arr[newy * sizx + newx] = old_arr[oldy * sizx + oldx];
}


//todo: carga en rectáncgulos al moverse y alejarse, carga especial para acercarse, usar interpolación?
void rellenar_pixels(complex_area area, complex Z0, int state) {
    
    clock_t t = clock();

    printf("TPF: %4lims\n", clock() - t);

    pre_gpu_pixel_funct <<< (area.surface->w / (BLOCK_SIDE) + 1) * (area.surface->h / (BLOCK_SIDE) + 1), BLOCK_SIDE * BLOCK_SIDE >>> (gpu_mem.main_arr, area.surface->w, area.surface->h, area.center, area.upp, Z0, state, 0, area.surface->h, 0, area.surface->w);

    cudaDeviceSynchronize();

    printf("TPF: %4lims\n", clock() - t);

    for(int i = BLOCK_SIDE / 2; i >= 1; i /= 2)
        gpu_pixel_funct <<<(area.surface->w / BLOCK_SIDE + 1) * (area.surface->h / BLOCK_SIDE + 1), BLOCK_SIDE * BLOCK_SIDE >>> (gpu_mem.main_arr, area.surface->w, area.surface->h, area.center, area.upp, Z0, state, i, 0, area.surface->h, 0, area.surface->w);

    cudaDeviceSynchronize();

    printf("TPF: %4lims\n", clock() - t);

    cudaMemcpy(area.surface->pixels, gpu_mem.main_arr, area.surface->w * area.surface->h * sizeof(int), cudaMemcpyDeviceToHost);

    printf("TPF: %4lims\n", clock() - t);

}

void transformar_area(complex_area last_calc, complex_area display) {

    cudaMemcpy(gpu_mem.main_arr, last_calc.surface->pixels, last_calc.surface->h * last_calc.surface->w * sizeof(int), cudaMemcpyHostToDevice);

    gpu_arr_transform <<<(last_calc.surface->w / BLOCK_SIDE + 1) * (last_calc.surface->h / BLOCK_SIDE + 1), BLOCK_SIDE* BLOCK_SIDE >>> (gpu_mem.main_arr, gpu_mem.copy_arr, last_calc.center, last_calc.upp, display.center, display.upp, last_calc.surface->w, last_calc.surface->h);

    cudaMemcpy(display.surface->pixels, gpu_mem.copy_arr, last_calc.surface->h * last_calc.surface->w * sizeof(int), cudaMemcpyDeviceToHost);
}

void draw_point_special(complex coords, int* arr, int sizx, int sizy, complex center, double units_per_pixel) {

    int x =  (coords.real - center.real) / units_per_pixel;
    int y = -(coords.imag - center.imag) / units_per_pixel;

    x += sizx / 2;
    y += sizy / 2;

    if (x >= 0 && x < sizx && y >= 0 && y < sizy) {
        for (int j = -POINT_RADIUS, i; j <= POINT_RADIUS; j++)
            for (i = -POINT_RADIUS; i <= POINT_RADIUS; i++)
                if (x + i >= 0 && x + i < sizx && y + j >= 0 && y + j < sizy)
                    arr[(y + j) * sizx + x + i] = 0xff0000ff;
    }
}

void draw_point(complex coords, int* arr, int sizx, int sizy, complex center, double units_per_pixel, double u) {

    int x =  (coords.real - center.real) / units_per_pixel;
    int y = -(coords.imag - center.imag) / units_per_pixel;

    x += sizx / 2;
    y += sizy / 2;

    unsigned char blue, green, red;

    blue = (0xff * (u < 0.25)) | (char)(0xff * (0.5 - u) * 4 * (0.25 <= u && u < 0.5));

    green = (char)(0xff * (u) * 4 * (u < 0.25)) | (0xff * (0.25 <= u && u < 0.75)) | (char)(0xff * (1 - u) * 4 * (0.75 <= u && u < 1));

    red = (char)(0xff * (u - 0.5) * 4 * (0.5 <= u && u < 0.75)) | (0xff * (0.75 <= u && u < 1));

    if (x >= 0 && x < sizx && y >= 0 && y < sizy) {
        for(int j = -POINT_RADIUS, i; j <= POINT_RADIUS; j++)
            for(i = - POINT_RADIUS; i <= POINT_RADIUS; i++)
                if (x + i >= 0 && x + i < sizx && y + j >= 0 && y + j < sizy) {
                    if ((i == -POINT_RADIUS || i == POINT_RADIUS) && (j == -POINT_RADIUS || j == POINT_RADIUS))
                        arr[(y + j) * sizx + x + i] = 0xffffffff; // rellenar color esquinas
                    else
                        arr[(y + j) * sizx + x + i] = 0xff | (blue << 8) | (green << 16) | (red << 24); // rellenar color normal
                }
    }
}

void draw_line(complex p1, complex p2, int* arr, int sizx, int sizy, complex center, double units_per_pixel) {

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

        arr[y * sizx + x] = 0xffffffff;
    }

    //todo: fix
}

void show_points(int *arr, int sizx, int sizy, complex center, double units_per_pixel) {
    if (point.len) {
        for (int i = 1; i < point.len; i++)
            draw_line(point.coords[i - 1], point.coords[i], arr, sizx, sizy, center, units_per_pixel);
        for(int i = 1; i < point.len; i++)
            draw_point(point.coords[i], arr, sizx, sizy, center, units_per_pixel, i / (double)point.len);
        draw_point_special(point.coords[0], arr, sizx, sizy, center, units_per_pixel);
        draw_point_special(Z0, arr, sizx, sizy, center, units_per_pixel);
    }
}

void calculate_points(int state) {
    complex C, Z;

    int k;

    if (state) {

        C = Z0;
        Z = point.coords[0];

    }
    else {

        C = point.coords[0];
        Z = Z0;

    }



    for (k = 1; k < SAMPLE_DEPTH; k++) {
        point.coords[k] = Z = sum_c(pow_c(Z, temp), C);
    }
  

    point.len = k;
}

int main(int argc, char* argv[]){

    SDL_Init(SDL_INIT_EVERYTHING);

    SDL_DisplayMode monitor_display;

    SDL_GetDesktopDisplayMode(0, &monitor_display);

    SDL_Window* window = SDL_CreateWindow("Fractals", 0, 0, monitor_display.w, monitor_display.h, SDL_WINDOW_BORDERLESS);

    SDL_SetWindowDisplayMode(window, &monitor_display);

    SDL_Surface* screen = SDL_GetWindowSurface(window);

    complex_area last_calculated_area;
    last_calculated_area.surface = SDL_CreateRGBSurfaceWithFormat(0, monitor_display.w, monitor_display.h, 32, SDL_PIXELFORMAT_RGBX8888);

    complex_area display_area;
    display_area.surface = SDL_CreateRGBSurfaceWithFormat(0, monitor_display.w, monitor_display.h, 32, SDL_PIXELFORMAT_RGBX8888);

    printf("ancho: %i\naltura: %i\n", monitor_display.w, monitor_display.h);

    cudaMalloc(&gpu_mem.main_arr,monitor_display.w * monitor_display.h * sizeof(int));

    cudaMalloc(&gpu_mem.copy_arr, monitor_display.w * monitor_display.h * sizeof(int));
    
    point.coords = (complex*)malloc(SAMPLE_DEPTH * sizeof(complex));

    last_calculated_area.center.real = cx_ini;
    last_calculated_area.center.imag = cy_ini;
    last_calculated_area.upp = upp_ini;

    
    display_area.center = last_calculated_area.center;
    display_area.upp = last_calculated_area.upp;
    

    int update_render = 1, updated_set = 0, state = 0, s_case = 1;

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
                    display_area.center.real += (mouse.posx - display_area.surface->w / 2) * display_area.upp * ZOOM;
                    display_area.center.imag -= (mouse.posy - display_area.surface->h / 2) * display_area.upp * ZOOM;
                    display_area.upp *= 1 - ZOOM;
                }
            if (local_zoom < 0)
                for (int i = 0; i >= local_zoom; i--) {
                    display_area.center.real -= (mouse.posx - display_area.surface->w / 2) * display_area.upp * ZOOM;
                    display_area.center.imag += (mouse.posy - display_area.surface->h / 2) * display_area.upp * ZOOM;
                    display_area.upp *= 1 + ZOOM;
                }

            //rect.len++;
            //update_render = 1;
        }

        
        //handle mouse movement
        if (mouse.button & (LEFT_CLICK)) {
            display_area.center.real -= mouse.deltax * display_area.upp;
            display_area.center.imag += mouse.deltay * display_area.upp;//cambio de signo necesario para mantener la dirección (matemáticamente)

            //update_render = 1;
            rect.len++;
        }

        
        if (mouse.button & MIDDLE_CLICK) {
            int x = mouse.posx - display_area.surface->w / 2;
            int y = -(mouse.posy - display_area.surface->h / 2);

            Z0.real = x * display_area.upp + display_area.center.real;
            Z0.imag = y * display_area.upp + display_area.center.imag;

            update_render = 1;
            updated_set = 1;
        }

        
        //put point
        if (mouse.button & (RIGTH_CLICK)) {
            //reasignar memoria
            free(point.coords);
            point.coords = (complex*)malloc(SAMPLE_DEPTH * sizeof(complex));

            //get primer pounto
            int x = mouse.posx - monitor_display.w / 2;
            int y = -(mouse.posy - monitor_display.h / 2);

            point.coords->real = x * display_area.upp + display_area.center.real;
            point.coords->imag = y * display_area.upp + display_area.center.imag;
            point.len = 1;

            //calcular el camino que sigue
            updated_set = 1;
        }

        if (keyboard['r']) {
            last_calculated_area.center.real = cx_ini;
            last_calculated_area.center.imag = cy_ini;
            last_calculated_area.upp = upp_ini;
            Z0 = { 0 };

            display_area.center = last_calculated_area.center;
            display_area.upp = last_calculated_area.upp;

            update_render = 1;
            updated_set = 1;
        }

        if (keyboard['s'] && s_case) {
            //todo state_change
            state = !state;
            s_case = 0;
            update_render = 1;
            updated_set = 1;
        }
        
        if (!keyboard['s'])
            s_case = 1;

        if (keyboard['f']) {
            last_calculated_area.center = display_area.center;
            last_calculated_area.upp = display_area.upp;
            SDL_LockSurface(last_calculated_area.surface);

            force_pixel_funct<<<(last_calculated_area.surface->w / BLOCK_SIDE + 1) * (last_calculated_area.surface->h / BLOCK_SIDE + 1), BLOCK_SIDE* BLOCK_SIDE >>>(gpu_mem.main_arr, last_calculated_area.surface->w, last_calculated_area.surface->h, last_calculated_area.center, last_calculated_area.upp, Z0, state, 0, last_calculated_area.surface->h, 0, last_calculated_area.surface->w);

            cudaMemcpy(last_calculated_area.surface->pixels, gpu_mem.main_arr, last_calculated_area.surface->w* last_calculated_area.surface->h * sizeof(int), cudaMemcpyDeviceToHost);

            SDL_UnlockSurface(last_calculated_area.surface);
        }

        if (updated_set) {
            calculate_points(state);
        }

        // write the pixels
        if (update_render) {
            SDL_LockSurface(last_calculated_area.surface);

            rellenar_pixels(last_calculated_area, Z0, state);
           
            SDL_UnlockSurface(last_calculated_area.surface);
        }
        SDL_LockSurface(last_calculated_area.surface);
        SDL_LockSurface(display_area.surface);
        transformar_area(last_calculated_area, display_area);
        SDL_UnlockSurface(display_area.surface);
        SDL_UnlockSurface(last_calculated_area.surface); // copiar superficie con transformación para ponerle los puntos
        


        SDL_LockSurface(display_area.surface);
        show_points((int*)display_area.surface->pixels, display_area.surface->w, display_area.surface->h, display_area.center, display_area.upp);
        SDL_UnlockSurface(display_area.surface);
        
        // copy to window
        SDL_BlitSurface(display_area.surface, NULL, screen, NULL);
        SDL_UpdateWindowSurface(window);
        update_render = 0;
        updated_set = 0;
    }

END:
    /*
    for (int i = 1; i < point.len; i++)
        printf("\n(%lf) {%5lf, %5lf} ^ {%5lf, %5lf} + {%5lf, %5lf} =", theta_host(point.coords[i]), point.coords[i].real, point.coords[i].imag, temp_real, temp_imag, point.coords[0].real, point.coords[0].imag);
        */

    printf("\n");

    SDL_Quit();

    SDL_FreeSurface(last_calculated_area.surface);
    SDL_FreeSurface(display_area.surface);
    SDL_DestroyWindow(window);
    free(point.coords);
    cudaFree(gpu_mem.main_arr);
    cudaFree(gpu_mem.copy_arr);
    return 0;
}