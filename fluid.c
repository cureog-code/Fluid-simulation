#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#define PI 3.14159265358979323846f
#define NX 300
#define NY 100
#define TIME_STEP 0.2f
#define SOLVE_ITERATIONS 120
#define DEFAULT_FRAMES 5000
#define HALF_SMOKE_HEIGHT 8.0f
#define INFLOW 1.25f
#define PERTUBATION_SCALE 0.025f
#define KINEMATIC_VISCOSITY 0.08f

// Tangential speed of the horizontal tunnel walls Positive values move the wall to the right, negative values move it left
#define BOTTOM_WALL_SPEED 0
#define TOP_WALL_SPEED 0

// Smoke source modes
#define SMOKE_MODE_PULSED_BAND 0
#define SMOKE_MODE_CONTINUOUS_STREAMS 1
#define SMOKE_SOURCE_MODE SMOKE_MODE_CONTINUOUS_STREAMS

// Parameters for SMOKE_MODE_CONTINUOUS_STREAMS, measured in grid cells
#define SMOKE_STREAM_SPACING 10
#define SMOKE_STREAM_THICKNESS 2
#define SMOKE_STREAM_DYE 1.0f

// Cylinder parameters
#define CYLINDER_RADIUS 15.0f

// Rectangle parameters
#define HALF_RECTANGLE_WIDTH 8.0f
#define HALF_RECTANGLE_HEIGHT 40.0f

// Ellipse parameters
#define SEMI_X_AXIS_ELLIPSE 15.0f
#define SEMI_Y_AXIS_ELLIPSE 8.0f

// NACA 00xx airfoil parameters, measured in grid cells/degrees
#define AIRFOIL_CENTER_X (0.38f * NX)
#define AIRFOIL_CENTER_Y (0.50f * NY)
#define AIRFOIL_CHORD 90.0f
#define AIRFOIL_THICKNESS_RATIO 0.12f
#define AIRFOIL_ANGLE_DEGREES 8.0f

// Venturi parameters
#define VENTURI_START_X (0.10f * NX)
#define VENTURI_END_X (0.70f * NX)
#define VENTURI_THROAT_HALF_HEIGHT (0.18f * NY)

// Cylinder-array parameters
#define CYLINDER_ARRAY_COLUMNS 4
#define CYLINDER_ARRAY_ROWS 3
#define CYLINDER_ARRAY_RADIUS 5.5f
#define CYLINDER_ARRAY_START_X (0.25f * NX)
#define CYLINDER_ARRAY_START_Y (0.25f * NY)
#define CYLINDER_ARRAY_SPACING_X (0.15f * NX)
#define CYLINDER_ARRAY_SPACING_Y (0.25f * NY)

enum obstacle{
  NONE,
  CYLINDER,
  RECTANGLE,
  ELLIPSE,
  NOZZLE_CYLINDER,
  AIRFOIL,
  VENTURI,
  CYLINDER_ARRAY
};

static float u[NY][NX + 1]; // Horizontal velocity at vertical edges
static float v[NY + 1][NX]; // Vertical velocity at horizontal edges
static float dye[NY][NX];

static float next_u[NY][NX + 1];
static float next_v[NY + 1][NX];
static float next_dye[NY][NX];

static float p[NY][NX];
static float divergence[NY][NX];

static unsigned char solid[NY][NX];
static enum obstacle scene = CYLINDER;

/*
Channel 0: smoke in air
Channel 1: vorticity
Channel 2: velocity magnitude
*/
static float frame[3][NY][NX];

static float clampf(float x, float lo, float hi){
    return (x < lo)? lo : ((x > hi)? hi : x);
}

static float bilinear_interpolate(const float *arr, int width, int height, int stride, float x, float y){
    x = clampf(x, 0, (float)(width - 1));
    y = clampf(y, 0, (float)(height - 1));
    int i = (int)x;
    int j = (int)y;

    int i1 = (i + 1 < width)? i + 1 : i;
    int j1 = (j + 1 < height)? j + 1 : j;
    float fx = x - i, fy = y - j;
    float a0 = arr[j * stride + i] * (1 - fx) + arr[j * stride + i1] * fx;
    float a1 = arr[j1 * stride + i] * (1 - fx) + arr[j1 * stride + i1] * fx;
    return a0 * (1 - fy) + a1 * fy;
}

static float sample_u(float x, float y){
    return bilinear_interpolate(&u[0][0], NX + 1, NY, NX + 1, x, y - 0.5f);
}
static float sample_v(float x, float y){
    return bilinear_interpolate(&v[0][0], NX, NY + 1, NX, x - 0.5f, y);
}
static float sample_dye(float x, float y){
    return bilinear_interpolate(&dye[0][0], NX, NY, NX, x - 0.5f, y - 0.5f);
}

// Define the obstacle =======================================================================================================
// Helpers
static float triangle_edge(float px, float py, float ax, float ay, float bx, float by){
    return (px - bx) * (ay - by) - (ax - bx) * (py - by);
}
static int point_in_triangle(float px, float py, float ax, float ay, float bx, float by, float cx, float cy){
    float d1 = triangle_edge(px, py, ax, ay, bx, by);
    float d2 = triangle_edge(px, py, bx, by, cx, cy);
    float d3 = triangle_edge(px, py, cx, cy, ax, ay);
    int hasNegative = (d1 < 0.0f) || (d2 < 0.0f) || (d3 < 0.0f);
    int hasPositive = (d1 > 0.0f) || (d2 > 0.0f) || (d3 > 0.0f);
    return !(hasNegative && hasPositive);
}
static int point_in_airfoil(float x, float y, float airfoil_cx, float airfoil_cy, float chord, float thickness_ratio, float angle){
    float cosine = cosf(angle);
    float sine = sinf(angle);
    float dx = x - airfoil_cx;
    float dy = y - airfoil_cy;
    float local_x = cosine * dx + sine * dy;
    float local_y = -sine * dx + cosine * dy;
    float chord_position = local_x / chord + 0.5f;
    if (chord_position < 0.0f || chord_position > 1.0f) return 0;
    float x2 = chord_position * chord_position;
    float x3 = x2 * chord_position;
    float x4 = x3 * chord_position;
    float half_thickness = 5.0f * thickness_ratio * chord * (
        0.2969f * sqrtf(chord_position) - 0.1260f * chord_position - 0.3516f * x2 + 0.2843f * x3 - 0.1036f * x4
    );
    return fabsf(local_y) <= half_thickness;
}

static void geometry(enum obstacle shape){
    const float cx = 50.0f, cy = NY * 0.5f;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < NX; ++i){
            float dx = i + 0.5f - cx;
            float dy = j + 0.5f - cy;
            switch (shape){
                case NONE:{
                    solid[j][i] = 0;
                    break;
                }

                case CYLINDER:{
                    solid[j][i] = (dx * dx + dy * dy < CYLINDER_RADIUS * CYLINDER_RADIUS);
                    break;
                }

                case RECTANGLE:{
                    solid[j][i] = (fabsf(dx) < HALF_RECTANGLE_WIDTH && fabsf(dy) < HALF_RECTANGLE_HEIGHT);
                    break;
                }

                case ELLIPSE:{
                    solid[j][i] = (dx * dx / (SEMI_X_AXIS_ELLIPSE * SEMI_X_AXIS_ELLIPSE) + dy * dy / (SEMI_Y_AXIS_ELLIPSE * SEMI_Y_AXIS_ELLIPSE) < 1.0f);
                    break;
                }

                case NOZZLE_CYLINDER:{
                    float x = i + 0.5f;
                    float y = j + 0.5f;
                    int upperNozzle = point_in_triangle(x, y, 0.0f, NY, 0.0f, 0.74f * NY, 0.27f * NX, 0.60f * NY);
                    int lowerNozzle = point_in_triangle(x, y, 0.0f, 0.0f, 0.0f, 0.26f * NY, 0.27f * NX, 0.40f * NY);
                    float cylinderDx = x - 0.43f * NX;
                    float cylinderDy = y - 0.50f * NY;
                    int downstream_cylinder = cylinderDx * cylinderDx + cylinderDy * cylinderDy < CYLINDER_RADIUS * CYLINDER_RADIUS;
                    solid[j][i] = upperNozzle || lowerNozzle || downstream_cylinder;
                    break;
                }

                case AIRFOIL:{
                    float x = i + 0.5f;
                    float y = j + 0.5f;
                    float angle = AIRFOIL_ANGLE_DEGREES * PI / 180.0f;
                    solid[j][i] = point_in_airfoil(x, y, AIRFOIL_CENTER_X, AIRFOIL_CENTER_Y, AIRFOIL_CHORD, AIRFOIL_THICKNESS_RATIO, angle);
                    break;
                }

                case VENTURI:{
                float x = i + 0.5f;
                float y = j + 0.5f;
                float position = (x - VENTURI_START_X) / (VENTURI_END_X - VENTURI_START_X);
                if (position >= 0.0f && position <= 1.0f){
                    float contraction = sinf(PI * position);
                    contraction *= contraction;
                    float half_height = 0.5f * NY - (0.5f * NY - VENTURI_THROAT_HALF_HEIGHT) * contraction;
                    solid[j][i] = fabsf(y - cy) > half_height;
                }else{
                    solid[j][i] = 0;
                }
                break;
                }

                case CYLINDER_ARRAY:{
                    float x = i + 0.5f;
                    float y = j + 0.5f;
                    solid[j][i] = 0;
                    for (int column = 0; column < CYLINDER_ARRAY_COLUMNS; ++column){
                        float cylinder_x = CYLINDER_ARRAY_START_X + column * CYLINDER_ARRAY_SPACING_X;
                        float stagger = (column % 2)? 0.5f * CYLINDER_ARRAY_SPACING_Y : 0.0f;
                        for (int row = 0; row < CYLINDER_ARRAY_ROWS; ++row){
                            float cylinder_y = CYLINDER_ARRAY_START_Y + row * CYLINDER_ARRAY_SPACING_Y + stagger;
                            float arrayDx = x - cylinder_x;
                            float arrayDy = y - cylinder_y;
                            if (arrayDx * arrayDx + arrayDy * arrayDy < CYLINDER_ARRAY_RADIUS * CYLINDER_ARRAY_RADIUS){
                                solid[j][i] = 1;
                            }
                        }
                    }
                    break;
                }
            }
        }
    }
}

static void build_boundary(int step){
    const float t = step * TIME_STEP;

    // A weak lateral perturbation near the centre breaks exact mirror symmetry.
    const float pertubation = PERTUBATION_SCALE * sinf(2.0f * PI * 0.018f * t);
    for (int j = 0; j < NY; ++j){
        u[j][0] = INFLOW;
        u[j][NX] = u[j][NX-1]; // open outflow, might add reflective border later
        float d = (j + 0.5f - NY * 0.5f) / 19.0f;
        v[j][0] = pertubation * expf(-d * d);
    }
    for (int i = 0; i < NX; ++i){
        v[0][i] = v[NY][i] = 0;
    }
    for (int i = 0; i <= NX; ++i){
        u[0][i] = BOTTOM_WALL_SPEED;
        u[NY - 1][i] = TOP_WALL_SPEED;
    }
    for (int j = 0; j < NY; ++j){
        for (int i = 1; i < NX; ++i){
            if (solid[j][i - 1] || solid[j][i]) u[j][i] = 0;
        }
    }
    for (int j = 1; j < NY; ++j){
        for (int i = 0; i < NX; ++i){
            if (solid[j - 1][i] || solid[j][i]) v[j][i] = 0;
        }
    }
}

static void advect_velocity(void){
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i <= NX; ++i){
            float x = (float)i;
            float y = (float)j + 0.5f;
            float vx = sample_u(x, y);
            float vy = sample_v(x, y);
            float mx = x - 0.5f * TIME_STEP * vx;
            float my = y - 0.5f * TIME_STEP * vy;
            next_u[j][i] = sample_u(x - TIME_STEP * sample_u(mx, my), y - TIME_STEP * sample_v(mx, my));
        }
    }
    for (int j = 0; j <= NY; ++j){
        for (int i = 0; i < NX; ++i){
            float x = i + 0.5f, y = (float)j;
            float vx = sample_u(x, y), vy = sample_v(x, y);
            float mx = x - 0.5f * TIME_STEP * vx, my = y - 0.5f * TIME_STEP * vy;
            next_v[j][i] = sample_v(x - TIME_STEP * sample_u(mx, my), y - TIME_STEP * sample_v(mx, my));
        }
    }
    memcpy(u, next_u, sizeof(u));
    memcpy(v, next_v, sizeof(v));
}

static void diffuse_velocity(int step){
    const float diffusion_number = KINEMATIC_VISCOSITY * TIME_STEP;
    build_boundary(step);
    memcpy(next_u, u, sizeof(u));
    memcpy(next_v, v, sizeof(v));

    // Diffuse horizontal velocity
    for (int j = 1; j < NY - 1; ++j){
        for (int i = 1; i < NX; ++i){
            if (solid[j][i - 1] || solid[j][i]){
                next_u[j][i] = 0;
                continue;
            }
            float laplacian = u[j][i - 1] + u[j][i + 1] + u[j - 1][i] + u[j + 1][i] - 4.0f * u[j][i];
            next_u[j][i] = u[j][i] + diffusion_number * laplacian;
        }
    }

    // Diffuse vertical velocity
    for (int j = 1; j < NY; ++j){
        for (int i = 1; i < NX; ++i){
            if (solid[j - 1][i] || solid[j][i]){
                next_v[j][i] = 0;
                continue;
            }

            // At the open outlet, repeat the last value to impose dv/dx = 0.
            float right = (i + 1 < NX)? v[j][i + 1] : v[j][i];
            float laplacian = v[j][i - 1] + right + v[j - 1][i] + v[j + 1][i] - 4.0f * v[j][i];
            next_v[j][i] = v[j][i] + diffusion_number * laplacian;
        }
    }
    memcpy(u, next_u, sizeof(u));
    memcpy(v, next_v, sizeof(v));

    // Restore wall, inlet, outlet, and obstacle velocities after diffusion.
    build_boundary(step);
}

static void project(int step){
    build_boundary(step);

    // Compute divergence.
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < NX; ++i){
            divergence[j][i] = (solid[j][i])? 0.0f : (u[j][i + 1] - u[j][i] + v[j + 1][i] - v[j][i]) / TIME_STEP;
        }
    }

    // Solve the pressure Poisson equation.
    for (int iteration = 0; iteration < SOLVE_ITERATIONS; ++iteration){
        for (int j = 0; j < NY; ++j){
            for (int i = 0; i < NX; ++i){
                if (solid[j][i]) continue;

                float sum = 0.0f;
                int count = 0;

                // Left: Neumann at inlet or solid.
                if (i > 0 && !solid[j][i - 1]){
                    sum += p[j][i - 1];
                    ++count;
                }

                // Right: fluid neighbour or p = 0 outside outlet.
                if (i + 1 < NX){
                    if (!solid[j][i + 1]){
                        sum += p[j][i + 1];
                        ++count;
                    }
                }else{
                    ++count;
                }

                // Bottom: Neumann at wall or solid.
                if (j > 0 && !solid[j - 1][i]){
                    sum += p[j - 1][i];
                    ++count;
                }

                // Top: Neumann at wall or solid.
                if (j + 1 < NY && !solid[j + 1][i]){
                    sum += p[j + 1][i];
                    ++count;
                }

                if (count){
                    float target = (sum - divergence[j][i]) / count;
                    p[j][i] += 1.72f * (target - p[j][i]);
                }
            }
        }
    }

    // Apply the final pressure gradient to u exactly once.
    for (int j = 0; j < NY; ++j){
        for (int i = 1; i < NX; ++i){
            if (!solid[j][i - 1] && !solid[j][i]){
                u[j][i] -= TIME_STEP * (p[j][i] - p[j][i - 1]);
            }
        }
    }

    // Apply the final pressure gradient to v exactly once.
    for (int j = 1; j < NY; ++j){
        for (int i = 0; i < NX; ++i){
            if (!solid[j - 1][i] && !solid[j][i]){
                v[j][i] -= TIME_STEP * (p[j][i] - p[j - 1][i]);
            }
        }
    }
    build_boundary(step);
}

static float smoke_source_at_row(int j, int step){
    if (SMOKE_SOURCE_MODE == SMOKE_MODE_PULSED_BAND){
        float dy = fabsf(j + 0.5f - NY * 0.5f);
        float bands = (step % 36 < 24)? 1.0f : 0;
        float source = ((dy < HALF_SMOKE_HEIGHT)? 0.75f : 0) * bands;
        if (dy < 2.0f) source = 1.0f;
        return source;
    }

    // Continuous thin streams: always on in time, periodically separated along y from the bottom wall to the top wall.
    int phase = j % SMOKE_STREAM_SPACING;
    int first_dyed_row = (SMOKE_STREAM_SPACING - SMOKE_STREAM_THICKNESS) / 2;
    return (phase >= first_dyed_row && phase < first_dyed_row + SMOKE_STREAM_THICKNESS)? SMOKE_STREAM_DYE : 0;
}

static void advect_dye(int step){
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < NX; ++i){
            float x = i + 0.5f;
            float y = j + 0.5f;
            float vx = sample_u(x, y);
            float vy = sample_v(x, y);
            float mx = x - 0.5f * TIME_STEP * vx;
            float my = y - 0.5f * TIME_STEP * vy;
            next_dye[j][i] = (solid[j][i])? 0 : 0.999f * sample_dye(x - TIME_STEP * sample_u(mx, my), y - TIME_STEP * sample_v(mx, my));
        }
    }
    memcpy(dye, next_dye, sizeof(dye));
    for (int j = 0; j < NY; ++j){
        float source = smoke_source_at_row(j, step);
        for (int i = 2; i < 5; ++i){
            dye[j][i] = source;
        }
    }
}

static void make_frame(void){
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < NX; ++i){
            if (solid[j][i]){
                frame[0][j][i] = -1.0f;
                frame[1][j][i] = 0;
                frame[2][j][i] = 0;
            }else{
                frame[0][j][i] = dye[j][i];

                // dv/dx - du/dy, centred via MAC face averages.
                int il = (i > 0)? i - 1 : i, ir = (i + 1 < NX)? i + 1 : i;
                int jb = (j > 0)? j - 1 : j, jt = (j + 1 < NY)? j + 1 : j;
                float dv = (v[j][ir] + v[j + 1][ir] - v[j][il] - v[j + 1][il]) * 0.25f;
                float du = (u[jt][i] + u[jt][i + 1] - u[jb][i] - u[jb][i + 1]) * 0.25f;
                frame[1][j][i] = dv - du;
                float ux = 0.5f * (u[j][i] + u[j][i + 1]);
                float vy = 0.5f * (v[j][i] + v[j + 1][i]);
                frame[2][j][i] = sqrtf(ux * ux + vy * vy);
            }
        }
    }
}

// ====================================================================================================================================
// ====================================================================================================================================
int main(int argc, char **argv){
    #ifdef _WIN32
    _setmode(_fileno(stdout), _O_BINARY);
    #endif
    int frames = DEFAULT_FRAMES;
    for (int k = 1; k < argc; ++k){
        if (!strcmp(argv[k], "--frames") && (k + 1 < argc)) frames = atoi(argv[++k]);
        else if (!strcmp(argv[k], "--scene") && (k + 1 < argc)){
            const char *s = argv[++k];
            if (!strcmp(s, "cylinder")) scene = CYLINDER;
            else if (!strcmp(s, "rectangle")) scene = RECTANGLE;
            else if (!strcmp(s, "ellipse")) scene = ELLIPSE;
            else if (!strcmp(s, "nozzle")) scene = NOZZLE_CYLINDER;
            else if (!strcmp(s, "airfoil")) scene = AIRFOIL;
            else if (!strcmp(s, "venturi")) scene = VENTURI;
            else if (!strcmp(s, "cylinder-array")) scene = CYLINDER_ARRAY;
            else if (!strcmp(s, "none")) scene = NONE;
            else{
                fprintf(stderr, "Unknown scene: %s\n", s);
                return 1;
            }
        }else{
            fprintf(stderr, "Usage: %s [--frames N] [--scene cylinder|rectangle|ellipse|none]\n", argv[0]);
            return 1;
        }
    }
    if (frames < 1){
        fputs("--frames must be positive\n", stderr);
        return 1;
    }
    if (KINEMATIC_VISCOSITY < 0){
        fputs("KINEMATIC_VISCOSITY must be non-negative\n", stderr);
        return 1;
    }
    if (KINEMATIC_VISCOSITY * TIME_STEP > 0.25f){
        fprintf(stderr, "Explicit viscosity is unstable: viscosity * TIME_STEP = %.6f > 0.25\n", KINEMATIC_VISCOSITY * TIME_STEP);
        return 1;
    }
    geometry(scene);
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i <= NX; ++i){
            u[j][i] = INFLOW;
        }
    }
    build_boundary(0);

    // Header: four uint32: magic 'SMK1' (bytes), nx, ny, channel count.
    const unsigned char magic[4]={'S','M','K','1'};
    uint32_t dims[3]={NX, NY, 3};
    if (fwrite(magic, 1, 4, stdout) != 4 || fwrite(dims, 4, 3, stdout) != 3) return 1;
    for (int step = 0; step < frames; ++step){
        advect_velocity();
        diffuse_velocity(step);
        project(step);
        advect_dye(step);
        make_frame();
        if (fwrite(frame, sizeof(float), 3 * NY * NX, stdout) != (size_t)(3 * NY * NX)){
            fputs("Output pipe closed\n", stderr);
            return 1;
        }
    }
    fprintf(stderr, "\nSimulation completed: %d frames\n", frames);
    return (fflush(stdout) == 0)? 0 : 1;
}
// ====================================================================================================================================
// ====================================================================================================================================
