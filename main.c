/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
// ---- Networking (unused now) ----
/* #include "lwip/udp.h"
#include "lwip/pbuf.h"
#include "lwip/ip_addr.h"
#include "lwip/netif.h"
#include "ethernetif.h"   // for heth (optional for L2 blast)
extern struct netif gnetif;
static struct udp_pcb *g_udp;
static ip_addr_t g_dst_ip; */

// ---- AI runtime (X-CUBE-AI) ----
#include "ai_platform.h"
#include "ai_platform_interface.h"   // add this line (some packs need it)
#include "dcnn.h"
#include "dcnn_data.h"

// ---- DSP (CMSIS-DSP) ----
#include "arm_math.h"
//#include "arm_const_structs.h"


/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
#define AUDIO_BUFF_SIZE     960         // DMA half-transfer = 480 samples
#define BYTES_PER_SAMPLE    3
// Class indices must match training (CLASS_MAP in Python)
#define CLASS_IDX_N   0
#define CLASS_IDX_B   1   // your "B" class (you called it cage now)
#define CLASS_IDX_OR  2   // outer ring

/* #define UDP_HEADER_SIZE     8         // unused
#define UDP_PAYLOAD_SIZE    ((AUDIO_BUFF_SIZE / 2) * BYTES_PER_SAMPLE)
#define UDP_PACKET_SIZE     (UDP_HEADER_SIZE + UDP_PAYLOAD_SIZE) */

#define MIC_FULL_SCALE      (8388608.0f)     // 24-bit signed range divisor
#define SAMPLE_RATE_HZ      (32000U)

// Inference window / FFT
#define FFT_LEN             (8192U)          // 0.256 s @ 32 kHz
#define FFT_HALF            (FFT_LEN/2U)
#define FEATURE_LEN         (FFT_LEN)        // build 8192 features (mirrored spectrum)

// LED pulse duration
#define LED_PULSE_MS        (2000U)

// Class mapping: assume 3-class (0=normal, 1/2=faults). Adjust if needed.
#define CLASS_NORMAL        (0)

/* Ring/fill state for collecting 8192 samples */
typedef struct {
  uint32_t write;           // write index [0..FFT_LEN-1]
  uint8_t  full;            // window ready flag
} win_state_t;
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
#define MAX(a,b) ((a)>(b)?(a):(b))
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

SAI_HandleTypeDef hsai_BlockB2;
DMA_HandleTypeDef hdma_sai2_b;

/* USER CODE BEGIN PV */
// --- Audio DMA buffer: each word contains one 24-bit sample right-justified ---
uint32_t pAudBuf[AUDIO_BUFF_SIZE];

// --- FFT and feature buffers ---
static float32_t   fft_input[FFT_LEN];          // windowed time-domain, float32 [-1..1]
static float32_t   fft_workbuf[FFT_LEN];        // in-place workspace / RFFT output (complex interleaved)
static float32_t   fft_mag[FFT_HALF + 1];       // magnitude bins [0..N/2]
static int8_t      features_q7[FEATURE_LEN];    // quantized features (int8) for AI input

static arm_rfft_fast_instance_f32 rfft;

// --- Hamming coefficients (computed once) ---
static float32_t   win_hamming[FFT_LEN];

// --- AI handles/buffers (int8 model) ---
static ai_handle   net = AI_HANDLE_NULL;
static ai_u8       activations[AI_DCNN_DATA_ACTIVATIONS_SIZE];

static ai_i8       ai_in[AI_DCNN_IN_1_SIZE];     // must match FEATURE_LEN
static ai_i8       ai_out[AI_DCNN_OUT_1_SIZE];

// --- Windowing state ---
static win_state_t g_win = { .write = 0, .full = 0 };

// --- LED pulse ---
static uint8_t     led_active = 0;
static uint32_t    led_t_on_ms = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
// Sanity: prove LEDs can light


static void MX_DMA_Init(void);
static void MX_SAI2_Init(void);
/* USER CODE BEGIN PFP */
// static void Net_Init(void); // unused now

static void AI_Init(void);
static int  AI_Run_Inference(const int8_t* in, int8_t* out);
static void Build_Hamming(void);
static void Process_Window_and_Infer(void);
static void Convert_DMA_Chunk_to_Window(const uint32_t* src, uint32_t n);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

// -------------------- AI init --------------------
static void AI_Init(void)
{
  ai_error err;

  /* Create + bind activations in one call */
  err = ai_dcnn_create_and_init(&net,
                                (const ai_handle[]){ activations },  /* activations buffer(s) */
                                NULL);                                 /* optional params */
  if (err.type != AI_ERROR_NONE) {
    Error_Handler();
  }

  /* (Optional) sanity check input size vs our FEATURE_LEN */
  if (AI_DCNN_IN_1_SIZE != FEATURE_LEN) {
    Error_Handler();
  }
}


// One inference; returns predicted class index (argmax), or -1 on error
static int AI_Run_Inference(const int8_t* in, int8_t* out)
{
  ai_i32 n_batches;

  ai_buffer ai_input[AI_DCNN_IN_NUM] = {
    AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_S8, 1, 1, AI_DCNN_IN_1_SIZE, 1, (ai_handle)in)
  };
  ai_buffer ai_output[AI_DCNN_OUT_NUM] = {
    AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_S8, 1, 1, AI_DCNN_OUT_1_SIZE, 1, (ai_handle)out)
  };

  n_batches = ai_dcnn_run(net, &ai_input[0], &ai_output[0]);
  if (n_batches != 1) {
    (void)ai_dcnn_get_error(net);
    return -1;
  }

  // Argmax over S8 output (no dequant) — adjust if you require dequantization
  int best_idx = 0;
  int8_t best_v = out[0];
  for (int i = 1; i < (int)AI_DCNN_OUT_1_SIZE; ++i) {
    if (out[i] > best_v) { best_v = out[i]; best_idx = i; }
  }
  return best_idx;
}

// Build Hamming window once
static void Build_Hamming(void)
{
  for (uint32_t n = 0; n < FFT_LEN; ++n) {
    // Hamming (alpha=0.54, beta=0.46)
    win_hamming[n] = 0.54f - 0.46f * arm_cos_f32((2.0f * PI * n) / (FFT_LEN - 1));
  }
}

// Convert a DMA chunk (24-bit signed in 32-bit word) into float window buffer
static void Convert_DMA_Chunk_to_Window(const uint32_t* src, uint32_t n)
{
  // Append n samples into fft_input with wrapping until full window
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t w = src[i] & 0x00FFFFFFu;          // right-justified 24-bit
    // Sign-extend to 32-bit
    int32_t s = (int32_t)(w << 8) >> 8;
    // Normalize to [-1..1)
    float32_t v = ((float32_t)s) / MIC_FULL_SCALE;
    // Write into current slot
    fft_input[g_win.write] = v;
    g_win.write++;
    if (g_win.write >= FFT_LEN) {
      g_win.write = 0;
      g_win.full = 1;          // window ready
    }
  }
}

// Process full window: windowing → RFFT → magnitude → mirror → quantize → infer → LED
static void Process_Window_and_Infer(void)
{
  // 1) Apply Hamming window into work buffer
  for (uint32_t i = 0; i < FFT_LEN; ++i) {
    fft_workbuf[i] = fft_input[i] * win_hamming[i];
  }

  // 2) Real FFT
  arm_rfft_fast_f32(&rfft, fft_workbuf, fft_workbuf, 0 /* forward */);

  // 3) Compute magnitudes for bins 0..N/2 (RFFT output is complex interleaved)
  //    RFFT_fast places: bin 0 -> Re=out[0], Im=out[1]; bin k -> out[2k], out[2k+1]
  // bin 0 (DC) and Nyquist (N/2) are purely real in theory
  fft_mag[0] = fabsf(fft_workbuf[0]);  // DC magnitude (imag @ [1])
  for (uint32_t k = 1; k < FFT_HALF; ++k) {
    float32_t re = fft_workbuf[2u * k + 0u];
    float32_t im = fft_workbuf[2u * k + 1u];
    fft_mag[k] = sqrtf(re*re + im*im);
  }
  // Nyquist bin at k = N/2 is stored at out[1]???  For arm_rfft_fast_f32, length is N,
  // and the Nyquist real value is at out[1] (paired with DC). Many users instead set it 0.
  // We take a safe approach:
  fft_mag[FFT_HALF] = 0.0f;

  // 4) Build FEATURE_LEN = 8192 by mirroring magnitudes:
  // [0, 1, ..., FFT_HALF, FFT_HALF-1, ..., 1]
  // Also perform simple AGC scaling to int8
  // Find peak for scaling (avoid DC if you wish; here we include all)
  float32_t peak = 1e-9f;
  for (uint32_t k = 0; k <= FFT_HALF; ++k) {
    peak = MAX(peak, fft_mag[k]);
  }
  // Scale so that 'peak' -> 127
  float32_t scale = (peak > 0.0f) ? (127.0f / peak) : 1.0f;

  uint32_t idx = 0;
  for (uint32_t k = 0; k <= FFT_HALF; ++k) {
    float32_t q = fft_mag[k] * scale;
    int32_t qi = (int32_t)lrintf(q);
    if (qi > 127) qi = 127;
    if (qi < -128) qi = -128;
    features_q7[idx++] = (int8_t)qi;
  }
  for (int32_t k = (int32_t)FFT_HALF - 1; k >= 1; --k) {
    float32_t q = fft_mag[(uint32_t)k] * scale;
    int32_t qi = (int32_t)lrintf(q);
    if (qi > 127) qi = 127;
    if (qi < -128) qi = -128;
    features_q7[idx++] = (int8_t)qi;
  }
  // idx should now be FEATURE_LEN (8192)

  // 5) Copy features into AI input and run inference
  for (uint32_t i = 0; i < FEATURE_LEN; ++i) ai_in[i] = features_q7[i];

  int pred = AI_Run_Inference(ai_in, ai_out);
	  if (pred >= 0) {
	    // Normal → BLUE (LD2)
	    // Outer ring → RED (LD3)
	    // Cage/"B" → BOTH
	    GPIO_PinState red   = GPIO_PIN_RESET;
	    GPIO_PinState blue  = GPIO_PIN_RESET;

	    if (pred == CLASS_IDX_N) {
	      blue = GPIO_PIN_SET;
	    } else if (pred == CLASS_IDX_OR) {
	      red = GPIO_PIN_SET;
	    } else if (pred == CLASS_IDX_B) {
	      red = GPIO_PIN_SET;
	      blue = GPIO_PIN_SET;
	    }

	    // Pulse LEDs for 0.5s
	    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, blue);
	    HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, red);
	    led_active = (red == GPIO_PIN_SET) || (blue == GPIO_PIN_SET);
	    if (led_active) led_t_on_ms = HAL_GetTick();


  }

  // Window consumed
  g_win.full = 0;
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */
  /* USER CODE END 1 */

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);  // BLUE on
  HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_SET);  // RED on
  HAL_Delay(1000);
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_RESET);
  MX_DMA_Init();
  MX_SAI2_Init();
  /* USER CODE BEGIN 2 */

  // --- Networking now unused ---
  // Net_Init();

  // Build Hamming window and init FFT
  Build_Hamming();
  if (arm_rfft_fast_init_f32(&rfft, FFT_LEN) != ARM_MATH_SUCCESS) {
    Error_Handler();
  }

  // Init AI
  AI_Init();

  // Start SAI DMA (mono, 24-bit in 32-bit slot)
  HAL_SAI_Receive_DMA(&hsai_BlockB2, (uint8_t*)pAudBuf, AUDIO_BUFF_SIZE);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    // If a full window is ready, process it
    if (g_win.full) {
      Process_Window_and_Infer();
    }

    // LED pulse end after 0.5 s

    if (led_active) {
      uint32_t dt = HAL_GetTick() - led_t_on_ms;
      if (dt >= LED_PULSE_MS) {
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_RESET);
        led_active = 0;
      }
    }


    // (Optional) sleep/yield
    // HAL_Delay(1);
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    // No other tasks
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 216;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief SAI2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_SAI2_Init(void)
{

  /* USER CODE BEGIN SAI2_Init 0 */
  /* USER CODE END SAI2_Init 0 */

  /* USER CODE BEGIN SAI2_Init 1 */
  /* USER CODE END SAI2_Init 1 */
  hsai_BlockB2.Instance = SAI2_Block_B;
  hsai_BlockB2.Init.Protocol = SAI_FREE_PROTOCOL;
  hsai_BlockB2.Init.AudioMode = SAI_MODEMASTER_RX;
  hsai_BlockB2.Init.DataSize = SAI_DATASIZE_24;
  hsai_BlockB2.Init.FirstBit = SAI_FIRSTBIT_MSB;
  hsai_BlockB2.Init.ClockStrobing = SAI_CLOCKSTROBING_FALLINGEDGE;
  hsai_BlockB2.Init.Synchro = SAI_ASYNCHRONOUS;
  hsai_BlockB2.Init.OutputDrive = SAI_OUTPUTDRIVE_DISABLE;
  hsai_BlockB2.Init.NoDivider = SAI_MASTERDIVIDER_ENABLE;
  hsai_BlockB2.Init.FIFOThreshold = SAI_FIFOTHRESHOLD_EMPTY;
  hsai_BlockB2.Init.AudioFrequency = SAI_AUDIO_FREQUENCY_32K;
  hsai_BlockB2.Init.SynchroExt = SAI_SYNCEXT_DISABLE;
  hsai_BlockB2.Init.MonoStereoMode = SAI_STEREOMODE;
  hsai_BlockB2.Init.CompandingMode = SAI_NOCOMPANDING;
  hsai_BlockB2.FrameInit.FrameLength = 64;
  hsai_BlockB2.FrameInit.ActiveFrameLength = 32;
  hsai_BlockB2.FrameInit.FSDefinition = SAI_FS_STARTFRAME;
  hsai_BlockB2.FrameInit.FSPolarity = SAI_FS_ACTIVE_LOW;
  hsai_BlockB2.FrameInit.FSOffset = SAI_FS_BEFOREFIRSTBIT;
  hsai_BlockB2.SlotInit.FirstBitOffset = 1;
  hsai_BlockB2.SlotInit.SlotSize = SAI_SLOTSIZE_32B;
  hsai_BlockB2.SlotInit.SlotNumber = 2;
  hsai_BlockB2.SlotInit.SlotActive = 0x00000001;
  if (HAL_SAI_Init(&hsai_BlockB2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN SAI2_Init 2 */
  /* USER CODE END SAI2_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA2_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA2_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA2_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA2_Stream1_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */
  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD3_Pin|LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, GPIO_PIN_6, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_Btn_Pin */
  GPIO_InitStruct.Pin = USER_Btn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_Btn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : RMII_MDC_Pin RMII_RXD0_Pin RMII_RXD1_Pin */
  GPIO_InitStruct.Pin = RMII_MDC_Pin|RMII_RXD0_Pin|RMII_RXD1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

  /*Configure GPIO pins : RMII_REF_CLK_Pin RMII_MDIO_Pin RMII_CRS_DV_Pin */
  GPIO_InitStruct.Pin = RMII_REF_CLK_Pin|RMII_MDIO_Pin|RMII_CRS_DV_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pins : PB0 RMII_TXD1_Pin */
  GPIO_InitStruct.Pin = GPIO_PIN_0|RMII_TXD1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : LD3_Pin LD2_Pin */
  GPIO_InitStruct.Pin = LD3_Pin|LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pins : STLK_RX_Pin STLK_TX_Pin */
  GPIO_InitStruct.Pin = STLK_RX_Pin|STLK_TX_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF7_USART3;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /*Configure GPIO pin : PG6 */
  GPIO_InitStruct.Pin = GPIO_PIN_6;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_OverCurrent_Pin */
  GPIO_InitStruct.Pin = USB_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : USB_SOF_Pin USB_ID_Pin USB_DM_Pin USB_DP_Pin */
  GPIO_InitStruct.Pin = USB_SOF_Pin|USB_ID_Pin|USB_DM_Pin|USB_DP_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF10_OTG_FS;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_VBUS_Pin */
  GPIO_InitStruct.Pin = USB_VBUS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_VBUS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : RMII_TX_EN_Pin RMII_TXD0_Pin */
  GPIO_InitStruct.Pin = RMII_TX_EN_Pin|RMII_TXD0_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  GPIO_InitStruct.Alternate = GPIO_AF11_ETH;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */
  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

// ---- Networking functions (unused; left commented for reference) ----
/*
void Net_Init(void)
{
  uint32_t t0 = HAL_GetTick();
  while (!netif_is_link_up(&gnetif) || !netif_is_up(&gnetif)) {
    MX_LWIP_Process();
    HAL_Delay(10);
    if (HAL_GetTick() - t0 > 3000) break;
  }
  ipaddr_aton("192.168.1.100", &g_dst_ip);
  g_udp = udp_new();
  if (!g_udp) Error_Handler();
  ip_set_option(g_udp, SOF_BROADCAST);
  udp_connect(g_udp, &g_dst_ip, 5004);
}
*/

// ---- DMA callbacks: fill FFT window from incoming audio ----
void HAL_SAI_RxHalfCpltCallback(SAI_HandleTypeDef *hsai)
{
  // First half: 480 samples
HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin); // heartbeat on half-transfer

  Convert_DMA_Chunk_to_Window(&pAudBuf[0], AUDIO_BUFF_SIZE/2);
}

void HAL_SAI_RxCpltCallback(SAI_HandleTypeDef *hsai)
{
  // Second half: 480 samples
  Convert_DMA_Chunk_to_Window(&pAudBuf[AUDIO_BUFF_SIZE/2], AUDIO_BUFF_SIZE/2);
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  __disable_irq();
  while (1)
  {
    // Trap
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  (void)file; (void)line;
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
