#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <float.h>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

// MACROS ============================================================================================================================
// ===================================================================================================================================
#define PI 3.14159265358979323846

// Size of the visible window in cells
#define NX 300
#define NY 90

// Physical domain
#define DOMAIN_LENGTH_X 450.0
#define DOMAIN_LENGTH_Y 135.0
#define DX (DOMAIN_LENGTH_X / (double)NX)
#define DY (DOMAIN_LENGTH_Y / (double)NY)
#define MAX_TIME_STEP 0.2
#define CFL_NUMBER 0.45
#define TIME_STEP_SAFETY 0.90 // Leave room for acceleration during the Runge-Kutta stages

// Momentum advection scheme, 1 is the unlimited third-order upwind-biased scheme and 0 is the Koren limited scheme
// The dye uses Koren reconstruction plus a conservative multidimensional flux limiter
#define MOMENTUM_UNLIMITED 1

// Pressure solve controls: PCG with a modified incomplete Cholesky preconditioner
#define PRESSURE_MAX_ITERATIONS 3200
#define PRESSURE_REL_TOLERANCE 1.0e-8
#define MIC_OMEGA 0.99
#define DIVERGENCE_RMS_TOLERANCE 1.0e-6
#define DIVERGENCE_MAX_TOLERANCE 2.0e-6
#define PROJECTION_MAX_CORRECTIONS 2
#define DYE_RANGE_TOLERANCE 1.0e-6
#define DEFAULT_FRAMES 10000
#define INFLOW 3.8f
#define PERTURBATION_SCALE 0.035
#define KINEMATIC_VISCOSITY 0.54

// The symmetry-breaking perturbation is applied during start-up
#define STARTUP_PERTURBATION_DURATION (60.0 * MAX_TIME_STEP)
#define INFLOW_RAMP_DURATION (60.0 * MAX_TIME_STEP)

// The output window is followed by an undamped downstream channel
#define OUTLET_BUFFER_CELLS ((2 * NX) / 5)
#define GRID_NX (NX + OUTLET_BUFFER_CELLS)
#define TIME_STEP_MAX_RETRIES 12
#define MIN_TIME_STEP 1.0e-6

// Tangential speeds of the horizontal tunnel walls
#define BOTTOM_WALL_SPEED 0.0
#define TOP_WALL_SPEED 0.0

// Smoke source modes
#define SMOKE_MODE_PULSED_BAND 0
#define SMOKE_MODE_CONTINUOUS_STREAMS 1
#define SMOKE_SOURCE_MODE SMOKE_MODE_CONTINUOUS_STREAMS
#define HALF_SMOKE_HEIGHT 8.0
#define SMOKE_PULSE_PERIOD (36.0 * MAX_TIME_STEP)
#define SMOKE_PULSE_ON_DURATION (24.0 * MAX_TIME_STEP)
#define DYE_RETENTION_PER_REFERENCE_STEP 0.999

// Parameters for SMOKE_MODE_CONTINUOUS_STREAMS, measured in physical units
#define SMOKE_STREAM_SPACING 10.0
#define SMOKE_STREAM_THICKNESS 2.0
#define SMOKE_STREAM_DYE 1.0

// Obstacle parameters ===============================================================================================================
// Obstacle center
#define OBSTACLE_CENTER_X 80.0
#define OBSTACLE_CENTER_Y (0.5 * DOMAIN_LENGTH_Y)

// Cylinder parameters
#define CYLINDER_RADIUS 16.0f

// Rectangle parameters
#define HALF_RECTANGLE_WIDTH 8.0
#define HALF_RECTANGLE_HEIGHT 40.0

// Ellipse parameters
#define SEMI_X_AXIS_ELLIPSE 15.0
#define SEMI_Y_AXIS_ELLIPSE 8.0

// Nozzle parameters
#define NOZZLE_LENGTH (0.3 * DOMAIN_LENGTH_X)
#define NOZZLE_INLET_HALF_HEIGHT (0.24 * DOMAIN_LENGTH_Y)
#define NOZZLE_CONVERGENCE 0.86
#define NOZZLE_CYLINDER_X (0.35 * DOMAIN_LENGTH_X)
#define NOZZLE_CYLINDER_Y (0.50 * DOMAIN_LENGTH_Y)

// NACA 00xx airfoil parameters
#define AIRFOIL_CENTER_X (0.38 * DOMAIN_LENGTH_X)
#define AIRFOIL_CENTER_Y (0.50 * DOMAIN_LENGTH_Y)
#define AIRFOIL_CHORD 90.0
#define AIRFOIL_THICKNESS_RATIO 0.12
#define AIRFOIL_ANGLE_DEGREES 8.0

// Venturi parameters
#define VENTURI_START_X (0.10 * DOMAIN_LENGTH_X)
#define VENTURI_END_X (0.70 * DOMAIN_LENGTH_X)
#define VENTURI_THROAT_HALF_HEIGHT (0.18 * DOMAIN_LENGTH_Y)

// Cylinder-array parameters
#define CYLINDER_ARRAY_COLUMNS 5
#define CYLINDER_ARRAY_ROWS 5
#define CYLINDER_ARRAY_RADIUS 2.5
#define CYLINDER_ARRAY_START_X (0.25 * DOMAIN_LENGTH_X)
#define CYLINDER_ARRAY_START_Y (0.25 * DOMAIN_LENGTH_Y)
#define CYLINDER_ARRAY_SPACING_X (0.15 * DOMAIN_LENGTH_X)
#define CYLINDER_ARRAY_SPACING_Y (0.25 * DOMAIN_LENGTH_Y)
// ===================================================================================================================================
// ===================================================================================================================================

// Obstacle shapes selectable with --scene
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

// Inlet velocity profiles selectable with --inlet
enum inlet_kind{
    INLET_UNIFORM,
    INLET_PARABOLIC,
    INLET_COUETTE
};

// CONTAINERS ========================================================================================================================
// ===================================================================================================================================
// Runtime settings, the defaults come from the constants above
static enum obstacle scene = CYLINDER;
static enum inlet_kind inletMode = INLET_UNIFORM;
static double inflowSpeed = INFLOW;
static double kinematicViscosity = KINEMATIC_VISCOSITY;
static double maximumTimeStep = MAX_TIME_STEP;
static double cflNumber = CFL_NUMBER;
static double startupPerturbationScale = PERTURBATION_SCALE;
static double inflowRampDuration = INFLOW_RAMP_DURATION;
static double bottomWallSpeed = BOTTOM_WALL_SPEED;
static double topWallSpeed = TOP_WALL_SPEED;

// Flow state on the grid
static double u[NY][GRID_NX + 1]; // Horizontal velocity at vertical edges
static double v[NY + 1][GRID_NX]; // Vertical velocity at horizontal edges
static double dye[NY][GRID_NX];

// Copies of the state at the start of a step and right-hand sides of the Runge-Kutta stages
static double initialU[NY][GRID_NX + 1];
static double initialV[NY + 1][GRID_NX];
static double rhsU[NY][GRID_NX + 1];
static double rhsV[NY + 1][GRID_NX];

static double initialDye[NY][GRID_NX];
static double stageDye[NY][GRID_NX]; // Temporarily holds the Euler approximation
static double dyeFluxX[NY][GRID_NX + 1]; // Upwind scheme
static double dyeFluxY[NY + 1][GRID_NX];
static double dyeLowFluxX[NY][GRID_NX + 1];
static double dyeLowFluxY[NY + 1][GRID_NX];
static double dyeIncreaseRatio[NY][GRID_NX];
static double dyeDecreaseRatio[NY][GRID_NX];

// Obstacle masks, the obstacle is fixed so the face masks are filled once in geometry
static unsigned char solid[NY][GRID_NX];
static unsigned char uBlocked[NY][GRID_NX + 1]; // Mask of blocked u faces
static unsigned char vBlocked[NY + 1][GRID_NX]; // Mask of blocked v faces

// Momentum fluxes are evaluated once per face and shared by the two cells touching it
static double fluxUx[NY][GRID_NX]; // Flux of u in x at the centre of cell i
static double fluxUy[NY + 1][GRID_NX]; // Flux of u in y at corners, row j + 1 holds corner j for j from -1 to NY - 1
static double fluxVx[NY][GRID_NX + 1]; // Flux of v in x at corners, column i + 1 holds corner i for i from -1 to GRID_NX - 1
static double fluxVy[NY][GRID_NX]; // Flux of v in y at the centre of cell j

// Pressure and PCG work fields use double precision
static double pressure[NY][GRID_NX];
static double initialPressure[NY][GRID_NX];
static double pressureRhs[NY][GRID_NX];
static double pressureResidual[NY][GRID_NX];
static double preconditionedResidual[NY][GRID_NX];
static double pressureDirection[NY][GRID_NX];
static double matrixDirection[NY][GRID_NX];
static double triangularTemporary[NY][GRID_NX];
static double pressureDiagonal[NY][GRID_NX];
static double preconditionerDiagonal[NY][GRID_NX];
static double preconditionerInverse[NY][GRID_NX]; // One over the pivot, a multiply replaces a slow and latency-bound divide

// Coefficients pre-multiplied by the inverse pivot so the triangular solves have a short dependency chain
static double forwardLeft[NY][GRID_NX]; // Link to the left cell divided by the pivot of this cell
static double forwardBottom[NY][GRID_NX]; // Link to the cell beneath divided by the pivot of this cell
static double backwardRight[NY][GRID_NX]; // Left link of the cell on the right divided by the pivot of this cell
static double backwardTop[NY][GRID_NX]; // Beneath link of the cell above divided by the pivot of this cell
static double preconditionerLeft[NY][GRID_NX];
static double preconditionerBottom[NY][GRID_NX];

static double boundaryTime;
static double minimumOpenArea;
static int fluidCells;

// Diagnostics printed to stderr when the run ends
static long pressureSolves;
static long pcgIterations;
static long rejectedAttempts;
static double lastDivergenceRms;
static double lastDivergenceMax;
static double maximumAcceptedCfl;
static double smallestAcceptedTimeStep = DBL_MAX;

// Output frame sent to the viewer, the SMK1 format stays float32
// Channel 0 is smoke, channel 1 is vorticity, channel 2 is horizontal velocity and channel 3 is vertical velocity
// Velocities are taken at cell centers
static float frame[4][NY][NX];
// ===================================================================================================================================
// ===================================================================================================================================

// Helpers ===========================================================================================================================
// Limit x to the closed range from low to high
static double clampd(double x, double low, double high){
    return fmin(fmax(x, low), high);
}

// Tell whether the u face at row j and column i is blocked
static int u_face_blocked(int j, int i){
    if (j < 0 || j >= NY || i < 0 || i > GRID_NX) return 1;
    return uBlocked[j][i];
}

// Tell whether the v face at row j and column i is blocked
static int v_face_blocked(int j, int i){
    if (j < 0 || j > NY || i < 0 || i >= GRID_NX) return 1;
    return vBlocked[j][i];
}

// Smooth ramp from 0 to 1 over the ramp duration
static double inlet_ramp(double simulationTime){
    if (inflowRampDuration <= 0) return 1;
    double position = clampd((simulationTime / inflowRampDuration), 0, 1);
    return position * position * (3.0 - 2.0 * position);
}

// Horizontal inlet velocity at height y for the selected inlet profile
static double inlet_velocity(double y, double simulationTime){
    double position = clampd(y / DOMAIN_LENGTH_Y, 0, 1);
    if (inletMode == INLET_COUETTE) return bottomWallSpeed + (topWallSpeed - bottomWallSpeed) * position;
    double inletSpeed = inflowSpeed * inlet_ramp(simulationTime);
    return (inletMode == INLET_PARABOLIC)? 6.0 * inletSpeed * position * (1 - position) : inletSpeed;
}

// Total volume flux entering through the open part of the inlet column
static double inlet_flux(double simulationTime){
    double flux = 0;
    for (int j = 0; j < NY; ++j){
        if (!solid[j][0]) flux += inlet_velocity((j + 0.5) * DY, simulationTime) * DY;
    }
    return flux;
}

// Length shared by two closed intervals, used to average the smoke source over the whole inlet face
static double interval_overlap(double low, double high, double sourceLow, double sourceHigh){
    return fmax(0, fmin(high, sourceHigh) - fmax(low, sourceLow));
}

// Integral of the periodic stream indicator from its origin to position
static double smoke_stream_integral(double position){
    double periods = floor(position / SMOKE_STREAM_SPACING);
    double phase = position - periods * SMOKE_STREAM_SPACING;
    return periods * SMOKE_STREAM_THICKNESS + clampd(phase, 0, SMOKE_STREAM_THICKNESS);
}

// Face-averaged dye concentration released into the flow at the inlet row j
static double smoke_source_at_row(int j, double simulationTime){
    double low = j * DY, high = (j + 1) * DY;
    if (SMOKE_SOURCE_MODE == SMOKE_MODE_PULSED_BAND){
        double center = 0.5 * DOMAIN_LENGTH_Y;
        double core = interval_overlap(low, high, center - 2.0, center + 2.0);
        double band = interval_overlap(low, high, center - HALF_SMOKE_HEIGHT, center + HALF_SMOKE_HEIGHT);
        double shared = interval_overlap(low, high, center - fmin(2.0, HALF_SMOKE_HEIGHT), center + fmin(2.0, HALF_SMOKE_HEIGHT));
        int pulseOn = fmod(simulationTime, SMOKE_PULSE_PERIOD) < SMOKE_PULSE_ON_DURATION;
        return (core + (pulseOn? 0.75 * (band - shared) : 0)) / DY;
    }
    double firstDyedPosition = 0.5 * (SMOKE_STREAM_SPACING - SMOKE_STREAM_THICKNESS);
    double dyedLength = smoke_stream_integral(high - firstDyedPosition) - smoke_stream_integral(low - firstDyedPosition);
    return SMOKE_STREAM_DYE * clampd(dyedLength / DY, 0, 1);
}

// Tell whether the point is inside the rotated NACA 00xx airfoil
static int point_in_airfoil(double x, double y){
    double angle = AIRFOIL_ANGLE_DEGREES * PI / 180.0;
    double cosine = cos(angle);
    double sine = sin(angle);
    double dx = x - AIRFOIL_CENTER_X;
    double dy = y - AIRFOIL_CENTER_Y;
    // Rotate into airfoil coordinates and normalise the chord position to the range from 0 to 1
    double localX = cosine * dx + sine * dy;
    double localY = -sine * dx + cosine * dy;
    double chordPosition = localX / AIRFOIL_CHORD + 0.5;
    if (chordPosition < 0 || chordPosition > 1) return 0;
    double x2 = chordPosition * chordPosition;
    double x3 = x2 * chordPosition;
    double x4 = x3 * chordPosition;
    // Standard NACA thickness polynomial
    double halfThickness = 5.0 * AIRFOIL_THICKNESS_RATIO * AIRFOIL_CHORD * (0.2969 * sqrt(chordPosition) 
                        - 0.1260 * chordPosition - 0.3516 * x2 + 0.2843 * x3 - 0.1036 * x4);
    return fabs(localY) <= halfThickness;
}

// Define the obstacle ===============================================================================================================
// Mark the solid cells of the chosen shape, remove sealed pockets, fill the face masks and check the tunnel stays open
static int geometry(enum obstacle shape){
    fluidCells = 0;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            double x = (i + 0.5) * DX;
            double y = (j + 0.5) * DY;
            double dx = x - OBSTACLE_CENTER_X;
            double dy = y - OBSTACLE_CENTER_Y;
            int blocked = 0;
            switch (shape){

                case NONE:{
                    blocked = 0;
                    break;
                }

                case CYLINDER:{
                    blocked = dx * dx + dy * dy < CYLINDER_RADIUS * CYLINDER_RADIUS;
                    break;
                }

                case RECTANGLE:{
                    blocked = fabs(dx) < HALF_RECTANGLE_WIDTH && fabs(dy) < HALF_RECTANGLE_HEIGHT;
                    break;
                }

                case ELLIPSE:{
                    blocked = (dx * dx / (SEMI_X_AXIS_ELLIPSE * SEMI_X_AXIS_ELLIPSE) + dy * dy / (SEMI_Y_AXIS_ELLIPSE * SEMI_Y_AXIS_ELLIPSE)) < 1;
                    break;
                }

                case NOZZLE_CYLINDER:{
                    // The nozzle walls close in linearly and a cylinder sits downstream of the exit
                    double exitHalfHeight = NOZZLE_INLET_HALF_HEIGHT * (1 - NOZZLE_CONVERGENCE);
                    double fluidHalfHeight = NOZZLE_INLET_HALF_HEIGHT + x / NOZZLE_LENGTH * (exitHalfHeight - NOZZLE_INLET_HALF_HEIGHT);
                    double cylinderDx = x - NOZZLE_CYLINDER_X;
                    double cylinderDy = y - NOZZLE_CYLINDER_Y;
                    int nozzleSolid = x < NOZZLE_LENGTH && fabs(y - OBSTACLE_CENTER_Y) >= fluidHalfHeight;
                    int downstreamCylinder = cylinderDx * cylinderDx + cylinderDy * cylinderDy < CYLINDER_RADIUS * CYLINDER_RADIUS;
                    blocked = nozzleSolid || downstreamCylinder;
                    break;
                }

                case AIRFOIL:{
                    blocked = point_in_airfoil(x, y);
                    break;
                }

                case VENTURI:{
                    // A sine-squared contraction narrows the tunnel to the throat height
                    double position = (x - VENTURI_START_X) / (VENTURI_END_X - VENTURI_START_X);
                    if (position >= 0 && position <= 1){
                        double contraction = sin(PI * position);
                        contraction *= contraction;
                        double halfHeight = 0.5 * DOMAIN_LENGTH_Y - (0.5 * DOMAIN_LENGTH_Y - VENTURI_THROAT_HALF_HEIGHT) * contraction;
                        blocked = fabs(y - OBSTACLE_CENTER_Y) > halfHeight;
                    }
                    break;
                }

                case CYLINDER_ARRAY:{
                    // Every second column is shifted by half a row spacing
                    for (int column = 0; column < CYLINDER_ARRAY_COLUMNS; ++column){
                        for (int row = 0; row < CYLINDER_ARRAY_ROWS; ++row){
                            double cylinderX = CYLINDER_ARRAY_START_X + column * CYLINDER_ARRAY_SPACING_X;
                            double stagger = (column % 2)? 0.5 * CYLINDER_ARRAY_SPACING_Y : 0;
                            double cylinderY = CYLINDER_ARRAY_START_Y + row * CYLINDER_ARRAY_SPACING_Y + stagger;
                            double arrayDx = x - cylinderX;
                            double arrayDy = y - cylinderY;
                            if (arrayDx * arrayDx + arrayDy * arrayDy < CYLINDER_ARRAY_RADIUS * CYLINDER_ARRAY_RADIUS){
                                blocked = 1;
                            }
                        }
                    }
                    break;
                }
            }
            // The downstream buffer is always open
            if (i >= NX) blocked = 0;
            solid[j][i] = (unsigned char)blocked;
        }
    }

    // Every fluid cell must connect to the outlet column
    // Breadth-first search is implemented
    static int fillQueue[NY * GRID_NX];
    static unsigned char reached[NY][GRID_NX];
    memset(reached, 0, sizeof(reached));
    int head = 0;
    int tail = 0;
    for (int j = 0; j < NY; ++j){
        if (!solid[j][GRID_NX - 1]){
            reached[j][GRID_NX - 1] = 1;
            fillQueue[tail++] = j * GRID_NX + GRID_NX - 1;
        }
    }
    while (head < tail){
        int index = fillQueue[head++];
        int j = index / GRID_NX;
        int i = index % GRID_NX;
        const int neighbourJ[4] = {j, j, j - 1, j + 1};
        const int neighbourI[4] = {i - 1, i + 1, i, i};
        for (int k = 0; k < 4; ++k){
            int nj = neighbourJ[k], ni = neighbourI[k];
            if (nj < 0 || nj >= NY || ni < 0 || ni >= GRID_NX) continue;
            if (solid[nj][ni] || reached[nj][ni]) continue;
            reached[nj][ni] = 1;
            fillQueue[tail++] = nj * GRID_NX + ni;
        }
    }
    fluidCells = 0;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (!solid[j][i] && !reached[j][i]){
                solid[j][i] = 1;
            }
            fluidCells += !solid[j][i];
        }
    }

    // A face is blocked when a solid cell touches it
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i <= GRID_NX; ++i){
            uBlocked[j][i] = (unsigned char)((i > 0 && solid[j][i - 1]) || (i < GRID_NX && solid[j][i]));
        }
    }
    for (int j = 0; j <= NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            vBlocked[j][i] = (unsigned char)((j > 0 && solid[j - 1][i]) || (j < NY && solid[j][i]));
        }
    }

    // The narrowest open column sets the fastest possible inflow speed used by the time step estimate
    minimumOpenArea = DBL_MAX;
    for (int i = 0; i <= GRID_NX; ++i){
        double openArea = 0;
        for (int j = 0; j < NY; ++j){
            if (!u_face_blocked(j, i)) openArea += DY;
        }
        minimumOpenArea = fmin(minimumOpenArea, openArea);
    }
    if (!fluidCells || minimumOpenArea <= 0) return 0;
    return 1;
}

// ===================================================================================================================================
// Set the velocity on every blocked face to zero
static void enforce_solid_faces(void){
    for (int j = 0; j < NY; ++j){
        for (int i = 1; i <= GRID_NX; ++i){
            if (u_face_blocked(j, i)) u[j][i] = 0;
        }
    }
    for (int j = 1; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (v_face_blocked(j, i)) v[j][i] = 0;
        }
    }
}

// Apply the inlet, the tunnel walls and the obstacle to the velocity field
static void build_boundary(void){
    for (int j = 0; j < NY; ++j){
        u[j][0] = (solid[j][0])? 0 : inlet_velocity((j + 0.5) * DY, boundaryTime);
    }
    for (int i = 0; i < GRID_NX; ++i){
        v[0][i] = 0;
        v[NY][i] = 0;
    }
    enforce_solid_faces();
}

// Value of u in a neighbouring cell for the Laplacian, including ghost values at walls
static double u_diffusion_neighbour(int j, int i, int neighbourJ, int neighbourI){
    if (neighbourJ < 0) return (2 * bottomWallSpeed - u[j][i]);
    if (neighbourJ >= NY) return (2 * topWallSpeed - u[j][i]);
    if (neighbourI < 0) return u[j][0];
    if (neighbourI > GRID_NX) return u[j][GRID_NX];
    if (!u_face_blocked(neighbourJ, neighbourI)) return u[neighbourJ][neighbourI];
    if (neighbourI != i || neighbourI == 0 || neighbourI == GRID_NX) return 0;
    return (solid[neighbourJ][neighbourI - 1] && solid[neighbourJ][neighbourI])? -u[j][i] : 0;
}

// Value of v in a neighbouring cell for the Laplacian, including ghost values at walls
static double v_diffusion_neighbour(int j, int i, int neighbourJ, int neighbourI){
    if (neighbourI < 0) return -v[j][i];
    if (neighbourI >= GRID_NX) return v[j][GRID_NX - 1];
    if (neighbourJ < 0 || neighbourJ > NY) return 0;
    if (!v_face_blocked(neighbourJ, neighbourI)) return v[neighbourJ][neighbourI];
    if (neighbourJ != j || neighbourJ == 0 || neighbourJ == NY) return 0;
    return (solid[neighbourJ - 1][neighbourI] && solid[neighbourJ][neighbourI])? -v[j][i] : 0;
}

// MUSCL reconstruction ==============================================================================================================
// Return the smallest magnitude when all three arguments share a sign and zero otherwise
static double minmod3(double a, double b, double c){
    if (a > 0 && b > 0 && c > 0) return fmin(a, fmin(b, c));
    if (a < 0 && b < 0 && c < 0) return fmax(a, fmax(b, c));
    return 0;
}

// Koren limiter
static double limited_slope(double left, double center, double right, int positiveFlow){
    double backward = center - left, forward = right - center;
    double centered = (positiveFlow)? (backward + 2 * forward) / 3 : (2 * backward + forward) / 3;
    return minmod3(2 * backward, centered, 2 * forward);
}

// Slope used for the momentum components
// The unlimited choice is the kappa = 1/3 scheme, its face value is third-order accurate and it keeps the extrema that the limiter clips
static double momentum_slope(double left, double center, double right, int positiveFlow){
#if MOMENTUM_UNLIMITED
    double backward = center - left, forward = right - center;
    return (positiveFlow)? (backward + 2 * forward) / 3 : (2 * backward + forward) / 3;
#else
    return limited_slope(left, center, right, positiveFlow);
#endif
}

// Read u at any index
static double read_u(const double *field, int j, int i, int referenceJ, int referenceI){
    if (j < 0) return 2 * bottomWallSpeed - read_u(field, -j - 1, i, referenceJ, referenceI);
    if (j >= NY) return 2 * topWallSpeed - read_u(field, 2 * NY - j - 1, i, referenceJ, referenceI);
    if (i < 0) return 2 * inlet_velocity((j + 0.5) * DY, boundaryTime) - read_u(field, j, -i, referenceJ, referenceI);
    if (i > GRID_NX) return field[j * (GRID_NX + 1) + GRID_NX];
    if (!u_face_blocked(j, i)) return field[j * (GRID_NX + 1) + i];
    if (i == 0 || i == GRID_NX) return 0;
    
    if (solid[j][i - 1] != solid[j][i]) return 0;
    if (j != referenceJ){
        int direction = (referenceJ > j)? 1 : -1;
        for (int k = j + direction, steps = 0; steps < 3; ++steps, k += direction){
            if (k < 0 || k >= NY) break;
            if (!u_face_blocked(k, i)){
                int mirrorJ = 2 * k - direction - j;
                if (!u_face_blocked(mirrorJ, i)) return -field[mirrorJ * (GRID_NX + 1) + i];
                break;
            }
            if (solid[k][i - 1] != solid[k][i]) break;
        }
    }else if (i != referenceI){
        int direction = (referenceI > i)? 1 : -1;
        for (int k = i + direction, steps = 0; steps < 3; ++steps, k += direction){
            if (k <= 0 || k >= GRID_NX) break;
            if (solid[j][k - 1] != solid[j][k]){
                int mirrorI = 2 * k - i;
                if (!u_face_blocked(j, mirrorI)) return -field[j * (GRID_NX + 1) + mirrorI];
                break;
            }
        }
    }
    return 0;
}

// Read v at any index
static double read_v(const double *field, int j, int i, int referenceJ, int referenceI){
    if (j < 0) return -read_v(field, -j, i, referenceJ, referenceI);
    if (j > NY) return -read_v(field, 2 * NY - j, i, referenceJ, referenceI);
    if (j == 0 || j == NY) return 0;
    if (i < 0) return -read_v(field, j, -i - 1, referenceJ, referenceI);
    if (i >= GRID_NX) return field[j * GRID_NX + GRID_NX - 1];
    if (!v_face_blocked(j, i)) return field[j * GRID_NX + i];

    if (solid[j - 1][i] != solid[j][i]) return 0;
    if (i != referenceI){
        int direction = (referenceI > i)? 1 : -1;
        for (int k = i + direction, steps = 0; steps < 3; ++steps, k += direction){
            if (k < 0 || k >= GRID_NX) break;
            if (!v_face_blocked(j, k)){
                int mirrorI = 2 * k - direction - i;
                if (!v_face_blocked(j, mirrorI)) return -field[j * GRID_NX + mirrorI];
                break;
            }
            if (solid[j - 1][k] != solid[j][k]) break;
        }
    }else if (j != referenceJ){
        int direction = (referenceJ > j)? 1 : -1;
        for (int k = j + direction, steps = 0; steps < 3; ++steps, k += direction){
            if (k <= 0 || k >= NY) break;
            if (solid[k - 1][i] != solid[k][i]){
                int mirrorJ = 2 * k - j;
                if (!v_face_blocked(mirrorJ, i)) return -field[mirrorJ * GRID_NX + i];
                break;
            }
        }
    }
    return 0;
}

// Reconstruction slope of u around node (j, i) along x when horizontal is set and along y otherwise
static double u_slope(int j, int i, int horizontal, int positiveFlow){
    int dj = (horizontal)? 0 : 1, di = (horizontal)? 1 : 0;
    double left = read_u(&u[0][0], j - dj, i - di, j, i);
    double center = read_u(&u[0][0], j, i, j, i);
    double right = read_u(&u[0][0], j + dj, i + di, j, i);
    // Ghost reconstruction at a staircase corner is not a smooth three-point stencil
    if (u_face_blocked(j - dj, i - di) || u_face_blocked(j, i) || u_face_blocked(j + dj, i + di)){
        return limited_slope(left, center, right, positiveFlow);
    }
    return momentum_slope(left, center, right, positiveFlow);
}

// Reconstruction slope of v around node (j, i) along x when horizontal is set and along y otherwise
static double v_slope(int j, int i, int horizontal, int positiveFlow){
    int dj = (horizontal)? 0 : 1, di = (horizontal)? 1 : 0;
    double left = read_v(&v[0][0], j - dj, i - di, j, i);
    double center = read_v(&v[0][0], j, i, j, i);
    double right = read_v(&v[0][0], j + dj, i + di, j, i);
    if (v_face_blocked(j - dj, i - di) || v_face_blocked(j, i) || v_face_blocked(j + dj, i + di)){
        return limited_slope(left, center, right, positiveFlow);
    }
    return momentum_slope(left, center, right, positiveFlow);
}

// Upwind-biased flux of u carried in x through the centre of cell i
static double u_flux_x(int j, int i){
    double carrier = 0.5 * (u[j][i] + u[j][i + 1]);
    double value = (carrier >= 0)? u[j][i] + 0.5 * u_slope(j, i, 1, 1) : u[j][i + 1] - 0.5 * u_slope(j, i + 1, 1, 0);
    return carrier * value;
}

// Upwind-biased flux of u carried in y through the corner between rows j and j + 1
static double u_flux_y(int j, int i){
    double carrier = 0.5 * (v[j + 1][i - 1] + v[j + 1][i]);
    if (carrier == 0) return 0;
    double value = (carrier >= 0)? read_u(&u[0][0], j, i, j + 1, i) + 0.5 * u_slope(j, i, 0, 1)
                              : read_u(&u[0][0], j + 1, i, j, i) - 0.5 * u_slope(j + 1, i, 0, 0);
    return carrier * value;
}

// Upwind-biased flux of v carried in x through the corner between columns i and i + 1
static double v_flux_x(int j, int i){
    // Prescribed v = 0 at the inlet
    if (i < 0) return 0;
    double carrier = 0.5 * (u[j - 1][i + 1] + u[j][i + 1]);
    if (carrier == 0) return 0;
    double value = (carrier >= 0)? read_v(&v[0][0], j, i, j, i + 1) + 0.5 * v_slope(j, i, 1, 1)
                              : read_v(&v[0][0], j, i + 1, j, i) - 0.5 * v_slope(j, i + 1, 1, 0);
    return carrier * value;
}

// Upwind-biased flux of v carried in y through the centre of cell j
static double v_flux_y(int j, int i){
    double carrier = 0.5 * (v[j][i] + v[j + 1][i]);
    double value = (carrier >= 0)? v[j][i] + 0.5 * v_slope(j, i, 0, 1) : v[j + 1][i] - 0.5 * v_slope(j + 1, i, 0, 0);
    return carrier * value;
}

// Momentum right-hand side ==========================================================================================================
// Compute advection + viscous diffusion for u and v
static void velocity_rhs(double simulationTime){
    double horizontalWeight = 1 / (DX * DX);
    double verticalWeight = 1 / (DY * DY);
    memset(rhsU, 0, sizeof(rhsU));
    memset(rhsV, 0, sizeof(rhsV));
    double meanOutletVelocity = fabs(inlet_flux(boundaryTime) / DOMAIN_LENGTH_Y);

    // Evaluate each flux once and only where a neighbouring face needs it
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            fluxUx[j][i] = (!u_face_blocked(j, i) || !u_face_blocked(j, i + 1))? u_flux_x(j, i) : 0;
        }
    }
    for (int j = -1; j < NY; ++j){
        for (int i = 1; i < GRID_NX; ++i){
            fluxUy[j + 1][i] = (!u_face_blocked(j, i) || !u_face_blocked(j + 1, i))? u_flux_y(j, i) : 0;
        }
    }
    for (int j = 1; j < NY; ++j){
        fluxVx[j][0] = 0;
        for (int i = 0; i < GRID_NX; ++i){
            fluxVx[j][i + 1] = (!v_face_blocked(j, i) || !v_face_blocked(j, i + 1))? v_flux_x(j, i) : 0;
        }
    }
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            fluxVy[j][i] = (!v_face_blocked(j, i) || !v_face_blocked(j + 1, i))? v_flux_y(j, i) : 0;
        }
    }

    // Horizontal momentum
    // Interior faces use fluxes and a Laplacian, outlet faces use a convective condition
    for (int j = 0; j < NY; ++j){
        for (int i = 1; i < GRID_NX; ++i){
            if (u_face_blocked(j, i)) continue;
            double laplacian = horizontalWeight * (u_diffusion_neighbour(j, i, j, i - 1) + u_diffusion_neighbour(j, i, j, i + 1) - 2 * u[j][i])
                            + verticalWeight * (u_diffusion_neighbour(j, i, j - 1, i) + u_diffusion_neighbour(j, i, j + 1, i) - 2 * u[j][i]);
            rhsU[j][i] = -(fluxUx[j][i] - fluxUx[j][i - 1]) / DX
                         -(fluxUy[j + 1][i] - fluxUy[j][i]) / DY + kinematicViscosity * laplacian;
        }
        if (!u_face_blocked(j, GRID_NX)){
            double convectionSpeed = fmax(meanOutletVelocity, fabs(u[j][GRID_NX]));
            double laplacian = verticalWeight * (u_diffusion_neighbour(j, GRID_NX, j - 1, GRID_NX)
                                             + u_diffusion_neighbour(j, GRID_NX, j + 1, GRID_NX) - 2 * u[j][GRID_NX]);
            rhsU[j][GRID_NX] = -convectionSpeed * (u[j][GRID_NX] - u[j][GRID_NX - 1]) / DX + kinematicViscosity * laplacian;
        }
    }

    // Vertical momentum
    for (int j = 1; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (v_face_blocked(j, i)) continue;
            double laplacian = horizontalWeight * (v_diffusion_neighbour(j, i, j, i - 1) + v_diffusion_neighbour(j, i, j, i + 1) - 2 * v[j][i])
                            + verticalWeight * (v_diffusion_neighbour(j, i, j - 1, i) + v_diffusion_neighbour(j, i, j + 1, i) - 2 * v[j][i]);
            rhsV[j][i] = -(fluxVx[j][i + 1] - fluxVx[j][i]) / DX -(fluxVy[j][i] - fluxVy[j - 1][i]) / DY + kinematicViscosity * laplacian;
        }
    }

    // A short Gaussian push at the inlet to break the symmetry
    if (simulationTime < STARTUP_PERTURBATION_DURATION && startupPerturbationScale != 0){
        double remaining = clampd(1 - (simulationTime / STARTUP_PERTURBATION_DURATION), 0, 1);
        for (int j = 1; j < NY; ++j){
            if (!v_face_blocked(j, 0)){
                double distance = (j * DY - 0.5 * DOMAIN_LENGTH_Y) / 19.0;
                rhsV[j][0] += startupPerturbationScale * remaining * exp(-distance * distance) / MAX_TIME_STEP;
            }
        }
    }
}

// Conservative dye fluxes ===========================================================================================================
// Read a dye value with clamped indices, a solid cell returns the value of the reference cell instead
static double read_dye(const double *field, int j, int i, int referenceJ, int referenceI){
    if (j < 0) j = 0;
    if (j >= NY) j = NY - 1;
    if (i < 0) i = 0;
    if (i >= GRID_NX) i = GRID_NX - 1;
    if (solid[j][i]) return field[referenceJ * GRID_NX + referenceI];
    return field[j * GRID_NX + i];
}

// Koren-limited slope of the dye around cell (j, i) along x when horizontal is set and along y otherwise
static double dye_slope(const double *field, int j, int i, int horizontal, int positiveFlow){
    int dj = (horizontal)? 0 : 1;
    int di = (horizontal)? 1 : 0;
    return limited_slope(read_dye(field, j - dj, i - di, j, i), field[j * GRID_NX + i], read_dye(field, j + dj, i + di, j, i), positiveFlow);
}

// Compute the dye flux through every open face using the current velocity
static void build_dye_fluxes(const double *field){
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i <= GRID_NX; ++i){
            double value = 0, lowValue = 0;
            if (!u_face_blocked(j, i)){
                double carrier = u[j][i];
                if (i == 0){
                    value = (carrier >= 0)? smoke_source_at_row(j, boundaryTime) : field[j * GRID_NX];
                    lowValue = value;
                }else if (i == GRID_NX){
                    // Backflow carries clear external fluid into the real outlet
                    value = (carrier >= 0)? field[j * GRID_NX + i - 1] + 0.5 * dye_slope(field, j, i - 1, 1, 1) : 0;
                    lowValue = (carrier >= 0)? field[j * GRID_NX + i - 1] : 0;
                }else{
                    value = (carrier >= 0)? field[j * GRID_NX + i - 1] + 0.5 * dye_slope(field, j, i - 1, 1, 1) 
                                         : field[j * GRID_NX + i] - 0.5 * dye_slope(field, j, i, 1, 0);
                    lowValue = (carrier >= 0)? field[j * GRID_NX + i - 1] : field[j * GRID_NX + i];
                }
                value *= carrier;
                lowValue *= carrier;
            }
            dyeFluxX[j][i] = value;
            dyeLowFluxX[j][i] = lowValue;
        }
    }
    for (int j = 0; j <= NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            double value = 0, lowValue = 0;
            if (j > 0 && j < NY && !v_face_blocked(j, i)){
                double carrier = v[j][i];
                value = (carrier >= 0)? field[(j - 1) * GRID_NX + i] + 0.5 * dye_slope(field, j - 1, i, 0, 1)
                                     : field[j * GRID_NX + i] - 0.5 * dye_slope(field, j, i, 0, 0);
                lowValue = (carrier >= 0)? field[(j - 1) * GRID_NX + i] : field[j * GRID_NX + i];
                value *= carrier;
                lowValue *= carrier;
            }
            dyeFluxY[j][i] = value;
            dyeLowFluxY[j][i] = lowValue;
        }
    }
}

// Limit the correction, each face keeps one flux shared by its two cells
// This is a multidimensional flux-corrected transport limiter with the physical bounds 0 and maximumDye
static int limit_dye_fluxes(const double *source, double dt, double decayRate, double maximumDye){
    double tolerance = DYE_RANGE_TOLERANCE * fmax(1, maximumDye);
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            dyeIncreaseRatio[j][i] = dyeDecreaseRatio[j][i] = 0;
            if (solid[j][i]) continue;
            double value = source[j * GRID_NX + i] - dt * ((dyeLowFluxX[j][i + 1] - dyeLowFluxX[j][i]) / DX
                                                        + (dyeLowFluxY[j + 1][i] - dyeLowFluxY[j][i]) / DY
                                                        + decayRate * source[j * GRID_NX + i]);
            // An invalid low-order state means the Euler CFL condition failed
            if (!isfinite(value) || value < -tolerance || value > maximumDye + tolerance) return 0;
            double left = dt * (dyeFluxX[j][i] - dyeLowFluxX[j][i]) / DX;
            double right = -dt * (dyeFluxX[j][i + 1] - dyeLowFluxX[j][i + 1]) / DX;
            double bottom = dt * (dyeFluxY[j][i] - dyeLowFluxY[j][i]) / DY;
            double top = -dt * (dyeFluxY[j + 1][i] - dyeLowFluxY[j + 1][i]) / DY;
            double increase = fmax(left, 0) + fmax(right, 0) + fmax(bottom, 0) + fmax(top, 0);
            double decrease = fmax(-left, 0) + fmax(-right, 0) + fmax(-bottom, 0) + fmax(-top, 0);
            dyeIncreaseRatio[j][i] = (increase > 0)? clampd((maximumDye - value) / increase, 0, 1) : 1;
            dyeDecreaseRatio[j][i] = (decrease > 0)? clampd(value / decrease, 0, 1) : 1;
        }
    }
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i <= GRID_NX; ++i){
            if (u_face_blocked(j, i)) continue;
            double correction = dyeFluxX[j][i] - dyeLowFluxX[j][i];
            double fraction = 1;
            if (i > 0){
                double leftRatio = (correction >= 0)? dyeDecreaseRatio[j][i - 1] : dyeIncreaseRatio[j][i - 1];
                fraction = fmin(fraction, leftRatio);
            }

            if (i < GRID_NX){
                double rightRatio = (correction >= 0)? dyeIncreaseRatio[j][i] : dyeDecreaseRatio[j][i];
                fraction = fmin(fraction, rightRatio);
            }
            dyeFluxX[j][i] = dyeLowFluxX[j][i] + fraction * correction;
        }
    }
    for (int j = 1; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (v_face_blocked(j, i)) continue;
            double correction = dyeFluxY[j][i] - dyeLowFluxY[j][i];
            double fraction = (correction >= 0)? fmin(dyeDecreaseRatio[j - 1][i], dyeIncreaseRatio[j][i])
                                               : fmin(dyeIncreaseRatio[j - 1][i], dyeDecreaseRatio[j][i]);
            dyeFluxY[j][i] = dyeLowFluxY[j][i] + fraction * correction;
        }
    }
    return 1;
}

static int dye_forward_euler(const double *source, double *destination, double dt, double decayRate){
    build_dye_fluxes(source);
    double maximumDye = (SMOKE_SOURCE_MODE == SMOKE_MODE_PULSED_BAND)? 1.0 : SMOKE_STREAM_DYE;
    if (!limit_dye_fluxes(source, dt, decayRate, maximumDye)) return 0;
    double tolerance = DYE_RANGE_TOLERANCE * fmax(1, maximumDye);
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            int index = j * GRID_NX + i;
            if (solid[j][i]){
                destination[index] = 0;
                continue;
            }
            double rhs = -(dyeFluxX[j][i + 1] - dyeFluxX[j][i]) / DX -(dyeFluxY[j + 1][i] - dyeFluxY[j][i]) / DY - decayRate * source[index];
            double value = source[index] + dt * rhs;
            if (!isfinite(value) || value < -tolerance || value > maximumDye + tolerance) return 0;

            // Derivations at the roundoff and projection tolerance are clipped
            destination[index] = clampd(value, 0, maximumDye);
        }
    }
    return 1;
}

// Pressure projection ===============================================================================================================
// The matrix is A = -D * G where D is the divergence and G is the gradient
// Prescribed normal wall and inlet velocities receive no pressure correction
// At the outlet p = 0 is imposed on the boundary face, giving a coefficient of 2 / DX^2

// Build the five-point matrix and its modified incomplete Cholesky factor, returns 0 when a pivot is not positive
static int build_pressure_matrix_with(double omega){
    const double horizontalWeight = 1.0 / (DX * DX);
    const double verticalWeight = 1.0 / (DY * DY);
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (solid[j][i]){
                preconditionerLeft[j][i] = preconditionerBottom[j][i] = 0;
                pressureDiagonal[j][i] = 1;
                preconditionerDiagonal[j][i] = 1;
                preconditionerInverse[j][i] = 1;
                continue;
            }

            double diagonal = 0;
            if (i > 0 && !solid[j][i - 1]) diagonal += horizontalWeight;
            if (i + 1 < GRID_NX && !solid[j][i + 1]) diagonal += horizontalWeight;
            if (i == GRID_NX - 1) diagonal += 2 * horizontalWeight;
            if (j > 0 && !solid[j - 1][i]) diagonal += verticalWeight;
            if (j + 1 < NY && !solid[j + 1][i]) diagonal += verticalWeight;
            if (diagonal <= 0) return 0;
            pressureDiagonal[j][i] = diagonal;

            // Incomplete Cholesky factorization is computed once for the fixed obstacle
            preconditionerLeft[j][i] = (i > 0 && !solid[j][i - 1])? -horizontalWeight / preconditionerDiagonal[j][i - 1] : 0;
            preconditionerBottom[j][i] = (j > 0 && !solid[j - 1][i])? -verticalWeight / preconditionerDiagonal[j - 1][i] : 0;
            double pivot = diagonal - preconditionerLeft[j][i] * preconditionerLeft[j][i] - preconditionerBottom[j][i] * preconditionerBottom[j][i];

            // Modified IC gives back the diagonal fill-in that IC(0) drops so that row sums of A are kept, scaled by omega
            double droppedFill = 0;
            if (i > 0 && !solid[j][i - 1] && j + 1 < NY && !solid[j + 1][i - 1]){
                droppedFill += preconditionerLeft[j][i] * (-verticalWeight / preconditionerDiagonal[j][i - 1]); // Link above the left cell
            }
            if (j > 0 && !solid[j - 1][i] && i + 1 < GRID_NX && !solid[j - 1][i + 1]){
                droppedFill += preconditionerBottom[j][i] * (-horizontalWeight / preconditionerDiagonal[j - 1][i]); // Link right of the cell beneath
            }
            pivot -= omega * droppedFill;
            if (pivot <= 1.0e-14 * diagonal) return 0;
            preconditionerDiagonal[j][i] = sqrt(pivot);
            preconditionerInverse[j][i] = 1.0 / preconditionerDiagonal[j][i];
        }
    }
    // Pre-multiply the links by the inverse pivots used in the forward and backward substitutions
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            forwardLeft[j][i] = preconditionerLeft[j][i] * preconditionerInverse[j][i];
            forwardBottom[j][i] = preconditionerBottom[j][i] * preconditionerInverse[j][i];
            backwardRight[j][i] = (i + 1 < GRID_NX)? preconditionerLeft[j][i + 1] * preconditionerInverse[j][i] : 0;
            backwardTop[j][i] = (j + 1 < NY)? preconditionerBottom[j + 1][i] * preconditionerInverse[j][i] : 0;
        }
    }
    return 1;
}

// When MIC fails, fall back to weaker modification and finally plain IC(0)
static int build_pressure_matrix(void){
    static const double omegas[] = {MIC_OMEGA, 0.97, 0.9, 0.5, 0.0};
    for (int k = 0; k < (int)(sizeof(omegas) / sizeof(omegas[0])); ++k){
        if (build_pressure_matrix_with(omegas[k])) return 1;
    }
    return 0;
}

// Multiply the pressure matrix with a field and return the dot product of the field with the result, summed in index order
static double apply_pressure_matrix(const double *field, double *result){
    const double horizontalWeight = 1.0 / (DX * DX);
    const double verticalWeight = 1.0 / (DY * DY);
    double product = 0;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            int index = j * GRID_NX + i;
            if (solid[j][i]){
                result[index] = field[index];
                product += field[index] * result[index];
                continue;
            }
            double value = pressureDiagonal[j][i] * field[index];
            if (i > 0 && !solid[j][i - 1]) value -= horizontalWeight * field[index - 1];
            if (i + 1 < GRID_NX && !solid[j][i + 1]) value -= horizontalWeight * field[index + 1];
            if (j > 0 && !solid[j - 1][i]) value -= verticalWeight * field[index - GRID_NX];
            if (j + 1 < NY && !solid[j + 1][i]) value -= verticalWeight * field[index + GRID_NX];
            result[index] = value;
            product += field[index] * value;
        }
    }
    return product;
}

// Forward substitution L y = source, the running left value stays in a register so each cell waits only for a multiply and a subtract
static void preconditioner_forward(const double *source){
    for (int j = 0; j < NY; ++j){
        // Left neighbour, its coefficient is zero at i = 0 and next to solid cells
        double previous = 0;
        for (int i = 0; i < GRID_NX; ++i){
            double value = source[j * GRID_NX + i] * preconditionerInverse[j][i];
            if (j > 0) value -= forwardBottom[j][i] * triangularTemporary[j - 1][i];
            value -= forwardLeft[j][i] * previous;
            triangularTemporary[j][i] = value;
            previous = value;
        }
    }
}

// Backward substitution L^T x = y and return the dot product of the source with x
static double preconditioner_backward(const double *source, double *destination){
    double product = 0;
    for (int j = NY - 1; j >= 0; --j){
        // Right neighbour
        double next = 0;
        for (int i = GRID_NX - 1; i >= 0; --i){
            int index = j * GRID_NX + i;
            double value = triangularTemporary[j][i] * preconditionerInverse[j][i];
            if (j + 1 < NY) value -= backwardTop[j][i] * destination[index + GRID_NX];
            value -= backwardRight[j][i] * next;
            destination[index] = value;
            next = value;
            product += source[index] * value;
        }
    }
    return product;
}

// Apply the full preconditioner and return the dot product of the source with the result
static double apply_preconditioner(const double *source, double *destination){
    preconditioner_forward(source);
    return preconditioner_backward(source, destination);
}

// Check the divergence residual against the absolute RMS, maximum limits, and against the relative limit
static int residual_within_tolerance(double dt, double rhsRms, double pressureResidualRms, double pressureResidualMax){
    double rmsLimit = DIVERGENCE_RMS_TOLERANCE / dt;
    double maximumLimit = DIVERGENCE_MAX_TOLERANCE / dt;
    int relativeConverged = rhsRms <= rmsLimit || pressureResidualRms <= PRESSURE_REL_TOLERANCE * rhsRms;
    return pressureResidualRms <= rmsLimit && pressureResidualMax <= maximumLimit && relativeConverged;
}

// Measure the current residual over the fluid cells and check it against the tolerances
static int residual_small_enough(double dt, double rhsRms){
    double squaredSum = 0;
    double largest = 0;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (solid[j][i]) continue;
            double value = pressureResidual[j][i];
            squaredSum += value * value;
            largest = fmax(largest, fabs(value));
        }
    }
    return residual_within_tolerance(dt, rhsRms, sqrt(squaredSum / fluidCells), largest);
}

// CG update of p and r, fused with the residual norms over fluid cells
// It also runs the forward substitution of the preconditioner because that only needs the fresh residual of its own cell
static void advance_pressure(double alpha, double *rms, double *maximum){
    double squaredSum = 0;
    double largest = 0;
    for (int j = 0; j < NY; ++j){
        double previous = 0;
        for (int i = 0; i < GRID_NX; ++i){
            pressure[j][i] += alpha * pressureDirection[j][i];
            pressureResidual[j][i] -= alpha * matrixDirection[j][i];
            double value = pressureResidual[j][i] * preconditionerInverse[j][i];
            if (j > 0) value -= forwardBottom[j][i] * triangularTemporary[j - 1][i];
            value -= forwardLeft[j][i] * previous;
            triangularTemporary[j][i] = value;
            previous = value;
            if (solid[j][i]) continue;
            double residual = pressureResidual[j][i];
            squaredSum += residual * residual;
            largest = fmax(largest, fabs(residual));
        }
    }
    *rms = sqrt(squaredSum / fluidCells);
    *maximum = largest;
}

// Recompute the residual from scratch to remove the drift of the recursive CG update
static void true_pressure_residual(void){
    apply_pressure_matrix(&pressure[0][0], &matrixDirection[0][0]);
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            pressureResidual[j][i] = pressureRhs[j][i] - matrixDirection[j][i];
        }
    }
}

// Preconditioned conjugate gradient solve of the pressure Poisson equation, warm-started from the previous pressure
static int solve_pressure_poisson(double dt, double rhsRms){
    true_pressure_residual();
    if (residual_small_enough(dt, rhsRms)) return 1;
    double residualProduct = apply_preconditioner(&pressureResidual[0][0], &preconditionedResidual[0][0]);
    memcpy(pressureDirection, preconditionedResidual, sizeof(pressureDirection));
    for (int iteration = 0; iteration < PRESSURE_MAX_ITERATIONS; ++iteration){
        ++pcgIterations;
        double denominator = apply_pressure_matrix(&pressureDirection[0][0], &matrixDirection[0][0]);
        if (!(denominator > 0) || !isfinite(denominator) || !(residualProduct > 0) || !isfinite(residualProduct)) break;
        double alpha = residualProduct / denominator;
        double residualRms, residualMax;
        // This also leaves the forward-substituted residual in triangularTemporary
        advance_pressure(alpha, &residualRms, &residualMax);
        if (residual_within_tolerance(dt, rhsRms, residualRms, residualMax)){
            // Check the true residual before accepting the pressure solution
            true_pressure_residual();
            if (residual_small_enough(dt, rhsRms)) return 1;
            residualProduct = apply_preconditioner(&pressureResidual[0][0], &preconditionedResidual[0][0]);
            memcpy(pressureDirection, preconditionedResidual, sizeof(pressureDirection));
            continue;
        }
        double nextResidualProduct = preconditioner_backward(&pressureResidual[0][0], &preconditionedResidual[0][0]);
        if (!(nextResidualProduct > 0) || !isfinite(nextResidualProduct)) break;
        double beta = nextResidualProduct / residualProduct;
        for (int j = 0; j < NY; ++j){
            for (int i = 0; i < GRID_NX; ++i){
                pressureDirection[j][i] = preconditionedResidual[j][i] + beta * pressureDirection[j][i];
            }
        }
        residualProduct = nextResidualProduct;
    }
    true_pressure_residual();
    return residual_small_enough(dt, rhsRms);
}

// Build the rhs of the Poisson equation from the divergence of the velocity and return its RMS
static double compute_divergence_rhs(double dt){
    double squaredSum = 0;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            pressureRhs[j][i] = (solid[j][i])? 0 : -((u[j][i + 1] - u[j][i]) / DX + (v[j + 1][i] - v[j][i]) / DY) / dt;
            squaredSum += pressureRhs[j][i] * pressureRhs[j][i];
        }
    }
    return sqrt(squaredSum / fluidCells);
}

// Subtract dt times the pressure gradient from the velocity, the outlet faces use the p = 0 boundary value
static void subtract_pressure_gradient(double dt){
    for (int j = 0; j < NY; ++j){
        for (int i = 1; i < GRID_NX; ++i){
            if (!u_face_blocked(j, i)) u[j][i] -= dt * (pressure[j][i] - pressure[j][i - 1]) / DX;
        }
        if (!solid[j][GRID_NX - 1]) u[j][GRID_NX] += 2 * dt * pressure[j][GRID_NX - 1] / DX;
    }
    for (int j = 1; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (!v_face_blocked(j, i)) v[j][i] -= dt * (pressure[j][i] - pressure[j - 1][i]) / DY;
        }
    }
}

// Measure the actual velocity after projection, independently of the recursive PCG residual
static int divergence_small_enough(void){
    double squaredSum = 0, largest = 0;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (solid[j][i]) continue;
            double divergence = (u[j][i + 1] - u[j][i]) / DX + (v[j + 1][i] - v[j][i]) / DY;
            if (!isfinite(divergence)) return 0;
            squaredSum += divergence * divergence;
            largest = fmax(largest, fabs(divergence));
        }
    }
    lastDivergenceRms = sqrt(squaredSum / fluidCells);
    lastDivergenceMax = largest;
    return lastDivergenceRms <= DIVERGENCE_RMS_TOLERANCE && lastDivergenceMax <= DIVERGENCE_MAX_TOLERANCE;
}

// Make the velocity divergence-free and check the result, a small correction removes any remaining solve error
static int project(double dt){
    ++pressureSolves;
    for (int correction = 0; correction < PROJECTION_MAX_CORRECTIONS; ++correction){
        double rhsRms = compute_divergence_rhs(dt);
        if (!isfinite(rhsRms) || !solve_pressure_poisson(dt, rhsRms)) return 0;
        subtract_pressure_gradient(dt);
        if (divergence_small_enough()) return 1;
        // A second solve uses only the small divergence left after the first correction
        memset(pressure, 0, sizeof(pressure));
    }
    return 0;
}
// ===================================================================================================================================

// Time-step control =================================================================================================================
// Largest sum of the cell-crossing rates in x and y, returns -1 when the velocity contains a non-finite value
static double maximum_advection_rate(void){
    double maximumRate = 0, maximumU = 0, maximumV = 0;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < GRID_NX; ++i){
            if (solid[j][i]) continue;
            if (!isfinite(u[j][i]) || !isfinite(u[j][i + 1]) || !isfinite(v[j][i]) || !isfinite(v[j + 1][i])) return -1;
            double rate = fmax(fabs(u[j][i]), fabs(u[j][i + 1])) / DX + fmax(fabs(v[j][i]), fabs(v[j + 1][i])) / DY;
            maximumRate = fmax(maximumRate, rate);
            maximumU = fmax(maximumU, fmax(fabs(u[j][i]), fabs(u[j][i + 1])));
            maximumV = fmax(maximumV, fmax(fabs(v[j][i]), fabs(v[j + 1][i])));
        }
    }
    return fmax(maximumRate, maximumU / DX + maximumV / DY);
}

// Upper bound of the diagonal of the diffusion operator
static double diffusion_diagonal_bound(void){
    double horizontalWeight = 1 / (DX * DX), verticalWeight = 1 / (DY * DY);
    return 2 * horizontalWeight + 2 * verticalWeight + 2 * fmax(horizontalWeight, verticalWeight);
}

// Choose the largest stable dt
static double calculate_time_step(double simulationTime){
    double maximumRate = maximum_advection_rate();
    if (maximumRate < 0) return -1;
    double anticipatedRate = fabs(inlet_flux(simulationTime + maximumTimeStep)) / minimumOpenArea / DX;
    for (int j = 0; j < NY; ++j){
        if (!solid[j][0]) anticipatedRate = fmax(anticipatedRate, fabs(inlet_velocity((j + 0.5) * DY, simulationTime + maximumTimeStep)) / DX);
    }
    maximumRate = fmax(maximumRate, anticipatedRate);
    double decayRate = -log(DYE_RETENTION_PER_REFERENCE_STEP) / MAX_TIME_STEP;
    double rate = maximumRate + 0.5 * kinematicViscosity * diffusion_diagonal_bound() + decayRate;
    return (rate > 1.0e-12)? fmin(maximumTimeStep, TIME_STEP_SAFETY * cflNumber / rate) : maximumTimeStep;
}

// Check that the new velocity still satisfies the CFL limit for the chosen dt
static int time_step_valid(double dt, double *stageCfl){
    double maximumRate = maximum_advection_rate();
    if (maximumRate < 0) return 0;
    double decayRate = -log(DYE_RETENTION_PER_REFERENCE_STEP) / MAX_TIME_STEP;
    double rate = maximumRate + 0.5 * kinematicViscosity * diffusion_diagonal_bound() + decayRate;
    *stageCfl = dt * rate;
    return *stageCfl <= cflNumber * 1.0001;
}

// Three-stage projected Runge-Kutta =================================================================================================
// Advance with SSPRK(3,3), every velocity stage is projected before it transports the dye
// The stage times are t, t + dt and t + dt / 2, including the time-dependent inlet and source
static double simulation_step(double simulationTime){
    static const double initialWeights[3] = {0, 0.75, 1.0 / 3.0};
    static const double stageWeights[3] = {1, 0.25, 2.0 / 3.0};
    static const double stageTimes[3] = {0, 1, 0.5};
    static const double nextTimes[3] = {1, 0.5, 1};
    double dt = calculate_time_step(simulationTime);
    if (!isfinite(dt) || dt < MIN_TIME_STEP) return -1;
    memcpy(initialU, u, sizeof(u));
    memcpy(initialV, v, sizeof(v));
    memcpy(initialDye, dye, sizeof(dye));
    memcpy(initialPressure, pressure, sizeof(pressure));
    double decayRate = -log(DYE_RETENTION_PER_REFERENCE_STEP) / MAX_TIME_STEP;
    for (int retry = 0; retry < TIME_STEP_MAX_RETRIES; ++retry){
        memcpy(u, initialU, sizeof(u));
        memcpy(v, initialV, sizeof(v));
        memcpy(dye, initialDye, sizeof(dye));
        memcpy(pressure, initialPressure, sizeof(pressure));
        int accepted = 1;
        double attemptMaximumCfl = 0;
        for (int stage = 0; stage < 3 && accepted; ++stage){
            boundaryTime = simulationTime + stageTimes[stage] * dt;
            build_boundary();
            velocity_rhs(boundaryTime);
            accepted = dye_forward_euler(&dye[0][0], &stageDye[0][0], dt, decayRate);
            if (!accepted) break;
            double initialWeight = initialWeights[stage], stageWeight = stageWeights[stage];
            for (int j = 0; j < NY; ++j){
                for (int i = 1; i <= GRID_NX; ++i){
                    u[j][i] = initialWeight * initialU[j][i] + stageWeight * (u[j][i] + dt * rhsU[j][i]);
                }
            }
            for (int j = 1; j < NY; ++j){
                for (int i = 0; i < GRID_NX; ++i){
                    v[j][i] = initialWeight * initialV[j][i] + stageWeight * (v[j][i] + dt * rhsV[j][i]);
                }
            }
            for (int j = 0; j < NY; ++j){
                for (int i = 0; i < GRID_NX; ++i){
                    dye[j][i] = (solid[j][i])? 0 : initialWeight * initialDye[j][i] + stageWeight * stageDye[j][i];
                }
            }
            boundaryTime = simulationTime + nextTimes[stage] * dt;
            build_boundary();
            double stageCfl = 0;
            accepted = project(stageWeight * dt) && time_step_valid(dt, &stageCfl);
            if (accepted) attemptMaximumCfl = fmax(attemptMaximumCfl, stageCfl);
        }
        if (accepted){
            smallestAcceptedTimeStep = fmin(smallestAcceptedTimeStep, dt);
            maximumAcceptedCfl = fmax(maximumAcceptedCfl, attemptMaximumCfl);
            return dt;
        }
        ++rejectedAttempts;
        dt *= 0.5;
        if (dt < MIN_TIME_STEP) break;
    }

    // Restore all evolving fields, including the pressure warm start, on a numerical failure
    memcpy(u, initialU, sizeof(u));
    memcpy(v, initialV, sizeof(v));
    memcpy(dye, initialDye, sizeof(dye));
    memcpy(pressure, initialPressure, sizeof(pressure));
    boundaryTime = simulationTime;
    build_boundary();
    return -1;
}

// Tell whether an output derivative can use the cell without sampling the zero-filled solid interior
static int fluid_cell(int j, int i){
    return j >= 0 && j < NY && i >= 0 && i < GRID_NX && !solid[j][i];
}

// Horizontal or vertical velocity at a fluid cell centre
static double cell_velocity(int j, int i, int horizontalComponent){
    return (horizontalComponent)? 0.5 * (u[j][i] + u[j][i + 1]) : 0.5 * (v[j][i] + v[j + 1][i]);
}

// Centred derivative in the fluid, second-order one-sided derivative at a wall when two neighbours are available
static double cell_velocity_derivative(int j, int i, int horizontalComponent, int horizontalDerivative){
    int dj = (horizontalDerivative)? 0 : 1, di = (horizontalDerivative)? 1 : 0;
    double spacing = (horizontalDerivative )? DX : DY;
    int leftOpen = fluid_cell(j - dj, i - di), rightOpen = fluid_cell(j + dj, i + di);
    if (leftOpen && rightOpen){
        return (cell_velocity(j + dj, i + di, horizontalComponent) - cell_velocity(j - dj, i - di, horizontalComponent)) / (2 * spacing);
    }
    if (!leftOpen && !rightOpen) return 0;
    int direction = (rightOpen)? 1 : -1;
    double center = cell_velocity(j, i, horizontalComponent);
    double first = cell_velocity(j + direction * dj, i + direction * di, horizontalComponent);
    if (fluid_cell(j + 2 * direction * dj, i + 2 * direction * di)){
        double second = cell_velocity(j + 2 * direction * dj, i + 2 * direction * di, horizontalComponent);
        return direction * (-3 * center + 4 * first - second) / (2 * spacing);
    }
    return direction * (first - center) / spacing;
}

// Convert the staggered state into the four cell-centred output channels
static void make_frame(void){
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i < NX; ++i){
            if (solid[j][i]){
                frame[0][j][i] = -1;
                frame[1][j][i] = frame[2][j][i] = frame[3][j][i] = 0;
                continue;
            }
            double dvdx = cell_velocity_derivative(j, i, 0, 1);
            double dudy = cell_velocity_derivative(j, i, 1, 0);
            frame[0][j][i] = (float)dye[j][i];
            frame[1][j][i] = (float)(dvdx - dudy);
            frame[2][j][i] = (float)cell_velocity(j, i, 1);
            frame[3][j][i] = (float)cell_velocity(j, i, 0);
        }
    }
}
// ===================================================================================================================================

// ===================================================================================================================================
// Parse a finite double
static int parse_double(const char *text, double *result){
    char *end;
    errno = 0;
    double value = strtod(text, &end);
    if (errno || end == text || *end || !isfinite(value)) return 1;
    *result = value;
    return 0;
}


// Parse a non-negative integer
static int parse_int(const char *text, int *result){
    char *end;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || end == text || *end || value < 0 || value > 100000000) return 1;
    *result = (int)value;
    return 0;
}

// Read the command line options as name and value pairs and check that all settings are in range
static int parse_arguments(int argc, char **argv, int *frames){
    static const char *sceneNames[] = {"none", "cylinder", "rectangle", "ellipse", "nozzle", "airfoil", "venturi", "cylinder-array"};
    for (int k = 1; k < argc; ++k){
        const char *argument = argv[k];
        if (k + 1 >= argc){
            fprintf(stderr, "error: option %s needs a value\n", argument);
            return 1;
        }

        const char *value = argv[++k];
        int validValue = 1;
        if (!strcmp(argument, "--scene")){
            validValue = 0;
            for (int index = 0; index < 8; ++index){
                if (!strcmp(value, sceneNames[index])){
                    scene = (enum obstacle)index;
                    validValue = 1;
                }
            }
        }else if (!strcmp(argument, "--inlet")){
            if (!strcmp(value, "uniform")){
                inletMode = INLET_UNIFORM;
            }else if (!strcmp(value, "parabolic")){
                inletMode = INLET_PARABOLIC;
            }else if (!strcmp(value, "couette")){
                inletMode = INLET_COUETTE;
            }else{
                validValue = 0;
            }
        }else if (!strcmp(argument, "--frames")){
            validValue = !parse_int(value, frames);
        }else if (!strcmp(argument, "--inflow")){
            validValue = !parse_double(value, &inflowSpeed);
        }else if (!strcmp(argument, "--viscosity")){
            validValue = !parse_double(value, &kinematicViscosity);
        }else if (!strcmp(argument, "--dt")){
            validValue = !parse_double(value, &maximumTimeStep);
        }else if (!strcmp(argument, "--cfl")){
            validValue = !parse_double(value, &cflNumber);
        }else if (!strcmp(argument, "--ramp")){
            validValue = !parse_double(value, &inflowRampDuration);
        }else if (!strcmp(argument, "--perturb")){
            validValue = !parse_double(value, &startupPerturbationScale);
        }else if (!strcmp(argument, "--wall-top")){
            validValue = !parse_double(value, &topWallSpeed);
        }else if (!strcmp(argument, "--wall-bottom")){
            validValue = !parse_double(value, &bottomWallSpeed);
        }else if (!strcmp(argument, "--sponge")){
            double disabledRate;
            validValue = !parse_double(value, &disabledRate) && !disabledRate;
        }else{
            fprintf(stderr, "error: unknown option %s\n", argument);
            return 1;
        }
        if (!validValue){
            fprintf(stderr, "error: invalid value '%s' for %s\n", value, argument);
            return 1;
        }
    }

    if (NX < 4 || NY < 4 || OUTLET_BUFFER_CELLS < 0 || *frames < 1 || inflowSpeed < 0 || kinematicViscosity < 0 || maximumTimeStep <= 0 || cflNumber <= 0
        || cflNumber > 0.5 || inflowRampDuration < 0 || TIME_STEP_SAFETY <= 0 || TIME_STEP_SAFETY > 1 || PROJECTION_MAX_CORRECTIONS < 1){
        fprintf(stderr, "error: a numeric option is out of range\n");
        return 1;
    }

    if (!(DYE_RETENTION_PER_REFERENCE_STEP > 0 && DYE_RETENTION_PER_REFERENCE_STEP <= 1)
        || SMOKE_STREAM_SPACING <= 0 || SMOKE_STREAM_THICKNESS < 0 || SMOKE_STREAM_THICKNESS > SMOKE_STREAM_SPACING || SMOKE_STREAM_DYE < 0
        || SMOKE_PULSE_PERIOD <= 0 || SMOKE_PULSE_ON_DURATION < 0 || SMOKE_PULSE_ON_DURATION > SMOKE_PULSE_PERIOD){
        fprintf(stderr, "error: invalid smoke constants\n");
        return 1;
    }

    return 0;
}

// MAIN ==============================================================================================================================
// ===================================================================================================================================
// Set up the obstacle and the initial flow, then stream the header and one frame per step to stdout
int main(int argc, char **argv){
    _Static_assert(sizeof(float) == 4 && sizeof(uint32_t) == 4, "SMK1 requires float32/uint32.");
    _Static_assert(NX >= 4 && NY >= 4 && OUTLET_BUFFER_CELLS >= 0, "Invalid grid dimensions.");
#ifdef _WIN32
    if (_setmode(_fileno(stdout), _O_BINARY) == -1) return 1;
#endif
    int frames = DEFAULT_FRAMES;
    if (parse_arguments(argc, argv, &frames) != 0) return 1;
    if (!geometry(scene)){
        // A minimum open area of zero means some column is completely blocked
        fprintf(stderr, "error: bad geometry (fluidCells = %d, minimumOpenArea = %g)\n", fluidCells, minimumOpenArea);
        return 1;
    }
    if (!build_pressure_matrix()){
        fprintf(stderr, "error: pressure matrix factorization failed\n");
        return 1;
    }
    // Start from the inlet velocity everywhere and project once so the initial field is divergence-free
    boundaryTime = 0;
    for (int j = 0; j < NY; ++j){
        for (int i = 0; i <= GRID_NX; ++i){
            u[j][i] = inlet_velocity((j + 0.5) * DY, 0);
        }
    }
    build_boundary();
    if (!project(maximumTimeStep)){
        fprintf(stderr, "error: initial pressure projection did not converge\n");
        return 1;
    }
    build_boundary();
    // The header holds the magic text, the grid size and the channel count
    const unsigned char magic[4] = {'S','M','K','1'};
    uint32_t dims[3] = {NX, NY, 4};
    if (fwrite(magic, 1, 4, stdout) != 4 || fwrite(dims, sizeof(uint32_t), 3, stdout) != 3) return 1;
    double simulationTime = 0;
    for (int step = 0; step < frames; ++step){
        double dt = simulation_step(simulationTime);
        if (dt <= 0){
            fprintf(stderr, "error: simulation failed at step %d, t = %g\n", step, simulationTime);
            return 1;
        }
        simulationTime += dt;
        make_frame();
        if (fwrite(frame, sizeof(float), (size_t)4 * NY * NX, stdout) != (size_t)4 * NY * NX) return 1;
    }
    fprintf(stderr, "fluid: %d steps, %.1f PCG iterations per projection, %ld rejected step attempts\n",
            frames, (double)pcgIterations / (double)pressureSolves, rejectedAttempts);
    double outletFlux = 0;
    for (int j = 0; j < NY; ++j) outletFlux += u[j][GRID_NX] * DY;
    double inletFlux = inlet_flux(simulationTime);
    double fluxImbalance = fabs(outletFlux - inletFlux) / fmax(fabs(inletFlux), 1.0e-12);
    fprintf(stderr, "fluid: t = %.9g, min dt = %.6g, max accepted CFL = %.6g\n",
            simulationTime, smallestAcceptedTimeStep, maximumAcceptedCfl);
    fprintf(stderr, "fluid: divergence RMS = %.3e, max = %.3e, relative flux imbalance = %.3e\n",
            lastDivergenceRms, lastDivergenceMax, fluxImbalance);
    return fflush(stdout) == (0)? 0 : 1;
}
// ===================================================================================================================================
// ===================================================================================================================================