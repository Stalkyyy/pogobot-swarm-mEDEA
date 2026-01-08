#include "pogobase.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>


/**
 * CONFIGURATION FOR POGOBOT-SPECIFIC NETWORK
 */

#define GENOME_SIZE 43              // Adapted for Pogobot Sensors
                                    // Inputs: 3 IR + 6 IMU + 1 energy + 1 bias = 11
                                    // Hidden: 3 neurons (11*3=33) + bias (3)
                                    // Output: 2 motors (3*2=6) = 42 + 1 sigma
#define HIDDEN_NEURONS 3
#define IR_SENSORS 3                // 3 photosensors (back, front-left, front-right)
#define IMU_INPUTS 6                // 3 accel + 3 gyro
#define INPUT_SIZE 11               // 3 IR + 6 IMU + 1 energy + 1 bias
#define OUTPUT_SIZE 2               // left motor, right motor
#define MAX_GENOME_LIST 50          // Reduced for memory constraints
#define GENERATION_LIFETIME 400     // Steps per generation
#define SIGMA_MIN 0.01f
#define SIGMA_MAX 0.3f
#define SIGMA_INIT 0.08f
#define ALPHA_UPDATE 0.10f


/**
 * GENOME STRUCTURE
 */
typedef struct {
    float weights[GENOME_SIZE];     // NN weights + sigma parameter
    uint32_t age;                   // Age for debugging
} genome_t;



/**
 * USERDATA STRUCTURE
 */

 typedef struct {
    // Active genome controlling the robot
    genome_t active_genome;     
    
    // List of imported genomes from neighbours
    genome_t genome_list[MAX_GENOME_LIST]; 
    uint16_t genome_list_size;
    
    // Generation management
    uint32_t generation_counter;
    uint32_t steps_in_generation;
    
    time_reference_t generation_timer;  // Timer for generation duration
    
    // Statistics
    uint32_t total_genomes_received;
    uint32_t total_generations_inactive;
    uint16_t active_agents_encountered;
    
    // IR communication state
    uint8_t last_tx_payload[384];
    uint16_t last_tx_size;

    // Activity state : 0 = inactive/dead, 1 = active
    uint8_t is_active;
} USERDATA;



DECLARE_USERDATA(USERDATA);
REGISTER_USERDATA(USERDATA);

float random_float(void);
float gaussian_random(void);
void genome_randomize(genome_t *g);
void genome_mutate(genome_t *parent, genome_t *child);
float sigmoid(float x);
void evaluate_network(genome_t *genome, float *inputs, uint16_t *motor_outputs);
void msg_rx_callback(message_t *msg);
void broadcast_genome(void);
void user_step_active(void);
void user_step_inactive(void);


/**
 * RANDOM NUMBER GENERATION
 */

static uint32_t seed = 12345;

float random_float(void) {
    seed = seed * 1103515245 + 12345;
    return ((seed / 65536) % 32768) / 32768.0f;
}

float gaussian_random(void) {
    float u1 = random_float();
    float u2 = random_float();
    if (u1 < 0.001f) u1 = 0.001f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

/**
 * GENOME OPERATIONS
 */

void genome_randomize(genome_t *g) {
    for (int i = 0; i < GENOME_SIZE - 1; i++) {
        g->weights[i] = (random_float() - 0.5f) * 2.0f;
    }
    g->weights[GENOME_SIZE - 1] = SIGMA_INIT;
    g->age = 0;
}

void genome_mutate(genome_t *parent, genome_t *child) {
    memcpy(child, parent, sizeof(genome_t));
    float sigma = parent->weights[GENOME_SIZE - 1];
    
    // Adaptive sigma mutation
    float sigma_mutation = sigma * ((random_float() < 0.5f) ? 
                                    (1.0f - ALPHA_UPDATE) : 
                                    (1.0f + ALPHA_UPDATE));
    sigma_mutation = (sigma_mutation < SIGMA_MIN) ? SIGMA_MIN : 
                     (sigma_mutation > SIGMA_MAX) ? SIGMA_MAX : 
                     sigma_mutation;
    child->weights[GENOME_SIZE - 1] = sigma_mutation;
    
    // Mutate network weights
    for (int i = 0; i < GENOME_SIZE - 1; i++) {
        child->weights[i] += gaussian_random() * sigma_mutation;
    }
    
    child->age = parent->age + 1;
}



/**
 * NEURAL NETWORK EVALUATION
 */

// Sigmoid activation function
float sigmoid(float x) {
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

void evaluate_network(genome_t *genome, float *inputs, uint16_t *motor_outputs) {
    float hidden[HIDDEN_NEURONS];
    float outputs[OUTPUT_SIZE];
    
    int weight_idx = 0;
    
    // Input to hidden layer
    for (int h = 0; h < HIDDEN_NEURONS; h++) {
        float sum = 0.0f;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += inputs[i] * genome->weights[weight_idx++];
        }
        hidden[h] = sigmoid(sum);
    }
    
    // Hidden to output layer
    for (int o = 0; o < OUTPUT_SIZE; o++) {
        float sum = 0.0f;
        for (int h = 0; h < HIDDEN_NEURONS; h++) {
            sum += hidden[h] * genome->weights[weight_idx++];
        }
        outputs[o] = sigmoid(sum);
    }
    
    // Convert outputs to motor speeds [0, 1023]
    motor_outputs[0] = (uint16_t)(outputs[0] * 1023.0f);  // Left motor
    motor_outputs[1] = (uint16_t)(outputs[1] * 1023.0f);  // Right motor
}



/**
 * MESSAGE HANDLING
 */

// Callback for receiving IR messages with genomes
void msg_rx_callback(message_t *msg) {
    // Check if this is a genome message (simple header check)
    if (msg->header.payload_length == sizeof(genome_t)) {
        if (mydata->genome_list_size < MAX_GENOME_LIST) {
            memcpy(&mydata->genome_list[mydata->genome_list_size],
                   msg->payload,
                   sizeof(genome_t));
            mydata->genome_list_size++;
            mydata->total_genomes_received++;
            mydata->active_agents_encountered++;
        }
    }
}

// Send active genome to neighbors
void broadcast_genome(void) {
    message_t msg;
    msg.header.payload_length = sizeof(genome_t);
    memcpy(msg.payload, &mydata->active_genome, sizeof(genome_t));
    
    // Send omnidirectional with low power
    pogobot_infrared_set_power(pogobot_infrared_emitter_power_oneThird);
    pogobot_infrared_sendLongMessage_omniGen(msg.payload, sizeof(genome_t));
}


/**
 * MEDEA STEPS (Active vs inactive)
 */

void user_step_active(void) {

    // ===== STEP 1: READ SENSORS =====

    // 1. Photosensors (IR proximity)
    float ir_back = pogobot_photosensors_read(0) / 4096.0f;
    float ir_front_left = pogobot_photosensors_read(1) / 4096.0f;
    float ir_front_right = pogobot_photosensors_read(2) / 4096.0f;
    
    // 2. IMU data
    float acc[3], gyro[3];
    pogobot_imu_read(acc, gyro);
    
    // Normalize accelerometer (±16g)
    for (int i = 0; i < 3; i++) {
        acc[i] = acc[i] / 16.0f;
        if (acc[i] > 1.0f) acc[i] = 1.0f;
        if (acc[i] < -1.0f) acc[i] = -1.0f;
    }
    
    // Normalize gyroscope (±2000°/s)
    for (int i = 0; i < 3; i++) {
        gyro[i] = gyro[i] / 2000.0f;
        if (gyro[i] > 1.0f) gyro[i] = 1.0f;
        if (gyro[i] < -1.0f) gyro[i] = -1.0f;
    }
    
    // 3. Battery voltage
    int16_t battery_mv = pogobot_battery_voltage_read();
    float energy_level = (battery_mv - 3000.0f) / 1500.0f;
    if (energy_level < 0.0f) energy_level = 0.0f;
    if (energy_level > 1.0f) energy_level = 1.0f;
    
    // Construct input vector for MLP
    float nn_inputs[INPUT_SIZE];
    nn_inputs[0] = ir_back;
    nn_inputs[1] = ir_front_left;
    nn_inputs[2] = ir_front_right;
    nn_inputs[3] = acc[0];
    nn_inputs[4] = acc[1];
    nn_inputs[5] = acc[2];
    nn_inputs[6] = gyro[0];
    nn_inputs[7] = gyro[1];
    nn_inputs[8] = gyro[2];
    nn_inputs[9] = energy_level;
    nn_inputs[10] = 1.0f;  // bias neuron
    

    // ===== STEP 2: EVALUATE NETWORK =====

    uint16_t motor_speeds[2];
    evaluate_network(&mydata->active_genome, nn_inputs, motor_speeds);
    

    // ===== STEP 3: CONTROL MOTORS =====

    pogobot_motor_power_set(motorL, motor_speeds[0]);
    pogobot_motor_power_set(motorR, motor_speeds[1]);
    
    // Set LED color based on genome age (visual feedback for ACTIVE state)
    uint8_t age_color = (mydata->active_genome.age % 128) + 64;
    pogobot_led_setColor(0, age_color, 255 - age_color);
    

    // ===== STEP 4: BROADCAST GENOME REGULARLY =====

    if (mydata->steps_in_generation % 10 == 0) {
        broadcast_genome();
    }
    

    // ===== STEP 5: CHECK GENERATION END =====

    mydata->steps_in_generation++;
    
    if (mydata->steps_in_generation >= GENERATION_LIFETIME) {
        // ===== GENERATION END: APPLY mEDEA ALGORITHM =====
        
        mydata->generation_counter++;
        mydata->steps_in_generation = 0;
        
        // Stop motors
        pogobot_motor_power_set(motorL, motorStop);
        pogobot_motor_power_set(motorR, motorStop);
        
        // Clear active genome
        memset(&mydata->active_genome, 0, sizeof(genome_t));
        
        // mEDEA CORE MECHANISM:
        if (mydata->genome_list_size > 0) {
            // Select random genome from imported list and mutate
            uint16_t selected_idx = (uint16_t)(random_float() * mydata->genome_list_size);
            if (selected_idx >= mydata->genome_list_size) {
                selected_idx = mydata->genome_list_size - 1;
            }
            
            genome_mutate(&mydata->genome_list[selected_idx], &mydata->active_genome);
            
            // Robot remains ACTIVE
            mydata->is_active = 1;
            
        } else {
            // NO GENOME RECEIVED - Robot becomes INACTIVE
            mydata->is_active = 0;
            mydata->total_generations_inactive++;
            
            #ifndef SIMULATOR
                printf(
                    "[R%d] INACTIVE at Gen %lu (no genomes received)\n",
                    pogobot_helper_getid(),
                    mydata->generation_counter
                );
            #endif
        }
        
        // Clear genome list for next generation
        mydata->genome_list_size = 0;
        mydata->active_agents_encountered = 0;
    }
}

void user_step_inactive(void) {
    pogobot_led_setColor(0, 0, 0);

    // Motors stopped
    pogobot_motor_power_set(motorL, motorStop);
    pogobot_motor_power_set(motorR, motorStop);

    // Increment step counter to check generation end
    mydata->steps_in_generation++;
    
    // Check if generation is over
    if (mydata->steps_in_generation >= GENERATION_LIFETIME) {
        mydata->generation_counter++;
        mydata->steps_in_generation = 0;

        if (mydata->genome_list_size > 0) {
            uint16_t selected_idx = (uint16_t)(random_float() * mydata->genome_list_size);
            if (selected_idx >= mydata->genome_list_size) {
                selected_idx = mydata->genome_list_size - 1;
            }
            genome_mutate(&mydata->genome_list[selected_idx], &mydata->active_genome);

            mydata->is_active = 1;

            #ifndef SIMULATOR
                printf(
                    "[R%d] REACTIVATED at Gen %lu\n", 
                    pogobot_helper_getid(), 
                    mydata->generation_counter
                );
            #endif
        } else {
            mydata->is_active = 0;
            mydata->total_generations_inactive++;
            #ifndef SIMULATOR
                if (mydata->generation_counter % 20 == 0) {
                    printf(
                        "[R%d] INACTIVE Gen %lu (total inactive: %lu)\n",
                        pogobot_helper_getid(),
                        mydata->generation_counter,
                        mydata->total_generations_inactive
                    );
                }
            #endif
        }

        // Clear genome list for next generation
        mydata->genome_list_size = 0;
        mydata->active_agents_encountered = 0;
    }
}



/**
 * MEDEA ALGORITHM
 */

void user_init(void) {
#ifndef SIMULATOR
    printf("mEDEA Initialization\n");
#endif
    
    // Initialize random seed based on robot ID
    seed = pogobot_helper_getRandSeed();
    
    // Initialize active genome randomly
    genome_randomize(&mydata->active_genome);
    
    // Initialize genome list
    mydata->genome_list_size = 0;
    mydata->generation_counter = 0;
    mydata->steps_in_generation = 0;
    mydata->total_genomes_received = 0;
    mydata->active_agents_encountered = 0;

    mydata->is_active = 1;
    
    // Initialize timers and algorithm
    pogobot_stopwatch_reset(&mydata->generation_timer);
    
    // Set main loop frequency
    main_loop_hz = 60;
    max_nb_processed_msg_per_tick = 1;
    msg_rx_fn = msg_rx_callback;
    msg_tx_fn = NULL;
    error_codes_led_idx = 3;
}


void user_step(void) {
    if (!mydata->is_active) {
        user_step_inactive();
    } else {
        user_step_active();
    }
}


int main(void) {
    pogobot_init();
#ifndef SIMULATOR
    printf("mEDEA Robot Started\n");
#endif
    pogobot_start(user_init, user_step);
    return 0;
}

// MODELINE "{{{1
// vim:expandtab:softtabstop=4:shiftwidth=4:fileencoding=utf-8
// vim:foldmethod=marker