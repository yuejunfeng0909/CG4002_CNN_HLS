#include "cnn_model.h"
#include <string.h>

void cnn_action_detection(
		FUNCTION_SELECT_BIT_WIDTH function_select,
		int user_number,
		float data[INPUT_DEPTH],
		float raw_output[DENSE_OUTPUT_NODES],
		float weights_and_bias[CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH]) {
#pragma HLS INTERFACE mode=s_axilite port=function_select
#pragma HLS INTERFACE mode=s_axilite port=user_number
#pragma HLS INTERFACE mode=m_axi depth=6 port=data offset=slave
#pragma HLS INTERFACE mode=m_axi port=raw_output depth=5 offset=slave
#pragma HLS INTERFACE mode=m_axi port=weights_and_bias depth=576 offset=slave
#pragma HLS interface s_axilite port=return

	// Initialize weights and bias
	static CNN_DTYPE CNN_weights[CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH][CNN_KERNEL_COUNT] = {-0.3506035, 0.1358752, 0.5473298, 0.6220097, 0.010363945, 0.7122301, -0.7401288, 0.32280797, 0.53081477, 0.31035107, 1.5485948, -7.639086e-06, 2.251494, 4.546894, 4.000283, 0.13023454, 1.2440063, -0.41097412, 0.101985, 0.39417425, 0.15495315, 0.5921003, -0.9878295, -1.0161685, 0.9723504, 0.4373468, 0.11065434, -0.29487464, 0.34707156, 0.73545504, 0.24033095, -1.6419932, -0.9971774, 1.0598942, 0.42021334, 0.4029356, -0.23495361, -0.29779524, 0.03459399, 0.12590683, 0.83044976, 1.2424446, -0.09205594, 0.10861059, 2.9726028, 2.6002104, 1.7317176, 2.8981526, -0.08526775, 2.6888325, -0.34280193, -0.034222946, -1.7859656, 1.087587, 0.18546952, -0.14879335, -0.3514689, 0.2719163, -0.77244085, 0.86989546, 0.5437259, -0.13070303, 0.14422457, 2.5933952, 0.5830914, -0.5722357, 0.43831086, -0.13713993, 0.036155965, 0.12241125, 1.6764205, -0.02958831, -0.55680686, 2.3510408, 0.08109384, -1.0908608, -0.60346407, 3.3737702, -1.4794471, 1.114336, -1.5298029, 1.0399233, 0.030192621, 0.35307664, 0.77919513, 1.1692864, -0.30494544, 2.7221978, -0.29458275, 0.84719527, -0.45006904, -0.07888324, 1.0670341, 0.6430576, -0.3143841, 1.5757363, -0.78365684, -0.5761155, 0.37021834, 0.77991015, -0.8574349, 0.53673846, -0.5929252, -0.4388993, 0.71123564, -1.2585275, -0.50401914, -0.9042128, 0.076401904, -1.2688773, 0.9910942, -1.1819823, 0.4605002, 0.7787085, -0.665184, -0.3035609, 1.5108987, 0.2613189, -0.08529561, -1.2838535, -1.0142293, 0.15065555, 1.5064414, 0.6594356, 1.241954, 0.939198, -0.58276904, 0.80426157, -0.34153274, -1.2494986, 1.3024904, 0.38025257, 0.814846, 1.1959648, -1.3551407, -0.34128475, 0.9103305, 0.07322672, -0.37196285, -1.882124, -0.06773475, -1.8386602, -0.20798919, 0.99066335, -0.1324987, 0.9997303, -0.09805242, 1.1358845, 0.25939256, -1.1511363, 1.6644912, 1.5550041, -0.44642675, -2.0764415, -0.29716972, 0.3526495, -0.7168549, -1.8653003, -0.7837976, 1.8365273, -0.92542964, -0.57389265, 0.25299242, -0.6114421, 1.1169484, -0.6998251, 0.29446465, -0.5198606, -1.4295664, 0.27767387, 2.0250466, -0.3120661, 0.5427372, -0.19376938, 1.1765934, 0.37649247, -0.95343626, 0.17629857, -1.59724, -1.6107018, -0.15404572, -1.5334055, 0.71207273, 0.12113631, 0.14034258, -0.9142133, 1.4040337, -1.5440699, -0.117548265, -0.99037, 1.639848, -1.30783, -0.42715675, 0.48428255, 0.27956057, 0.29978713, -0.47448435, 0.3725505, -0.33632308, 0.41496965, 0.23210527, 0.2098591, 1.1124094, -0.22413518, 1.1113113, 4.5676894, 4.734715, -0.14669155, 1.6416454, 0.31582215, -0.08594998, 0.54510725, 0.42739254, 0.44031796, -0.70326596, -0.38024914, 0.98575085, 0.21634437, -0.113769025, -0.45183063, -0.1625995, 0.60331744, 0.7289295, -0.7790848, -0.9644629, 1.392574, 0.35058886, 0.2766967, -0.753926, -0.55572855, -0.420094, 0.21595268, 0.6053411, 0.7168037, -0.27262783, -0.15925948, 2.7535903, 1.284404, 1.0149031, 2.6469073, 0.6295022, 2.0145795, -0.0682574, 0.2320491, -1.0086269, 1.3908292, -0.05799953, -0.9196083, -0.72016513, 0.35066274, -0.2918458, 0.3509271, 0.3128855, 0.15490699, 0.5325676, 1.8169031, 0.46200398, -0.77643174, 0.23426823, 0.23361564, -0.49919808, 0.1217148, 1.5363044, -0.014840295, -0.5212049, 2.7677693, 0.74793434, -0.7832514, -1.6045264, 1.3890203, -1.1567934, 1.8972727, -1.4028375, 0.8610481, 0.3100028, 0.24165587, -0.45830914, 0.75256485, -0.33585593, 2.2166831, -0.02536087, 0.95877004, 0.1054156, -0.0248301, 1.6189917, 0.9481848, -0.2057655, 2.294403, -1.0707113, -0.6263951, 0.32715908, 0.6437832, -0.64530605, 0.118724205, -0.3475914, -0.4551918, 0.9030782, -0.21540198, -1.0209157, -0.1612889, 0.022069162, -0.82458234, 0.17136088, -0.6157805, 0.048401024, 0.061288796, 0.24470435, 0.26382586, 1.1522819, 0.4365861, 0.5236289, -0.82704824, -0.6962941, 0.37043232, 2.0682857, 0.66906315, 0.42710766, 0.4967824, -0.5099744, 0.51160467, 0.057486754, -0.5152153, 0.7887476, 0.32476768, 0.3214991, 0.50577325, -1.4944873, 0.025652284, 0.4497945, 0.010818674, -1.3049093, -0.9909684, 0.1584218, -0.15128244, -0.42745784, 0.22759591, -0.11599462, -2.1664095, -0.004714667, 0.75618356, 0.60564625, -0.69005513, 1.4815217, 0.67546654, -0.13380614, -1.4019169, -0.06734533, -0.21909282, -0.00345855, -0.8479737, -0.63416964, 0.44609883, -1.2842525, 0.5894219, 0.17781411, -0.044797394, 0.6060357, -0.91620135, -1.181269, 0.29418144, 0.2821636, -0.62995136, 1.9376005, 0.47117513, 0.032918923, 0.6895252, 0.3335658, 0.08671902, -1.0731901, -0.97813505, -0.16418919, -0.8427477, -0.69764656, 0.2485253, -0.11061468, 0.033353355, 0.1977429, -0.9176212, 0.51093507, -1.5427396, 0.40883023, -1.0900445, 1.7770315, -1.4402201, -0.44569543, 0.39974284, 0.51739454, 0.23252594, -0.9975659, 0.44435233, -0.13903959, 0.81365687, -0.004954367, -0.02981155, 0.7534813, -0.20033841, 0.9802741, 3.878667, 4.8985076, -0.5986365, 2.4047592, 0.91650164, -0.16334994, 0.31152448, 0.6082732, 0.36572912, -0.79170567, -0.10226794, 0.776963, -0.06699673, 0.034258142, -0.5737377, -0.39127502, 0.34721363, 0.54853606, -0.2669368, -0.71917444, 1.7681159, -0.46456334, -0.23970936, -0.96159816, -0.9728838, -0.33845016, 0.028594831, 0.8010312, 0.09889112, -0.6411067, -0.20249364, 1.8168731, 0.75824094, 0.40336534, 2.067512, 1.0921577, 1.343328, 0.23271042, 0.61405337, -0.6503495, 1.5963105, 0.13817643, -1.5005052, -0.8262539, 0.49260306, -0.31753546, 0.53780174, -0.16517289, 0.6722304, 0.2959463, 0.64491254, 0.64042926, -0.62084454, 0.13111296, 0.74781287, -0.32319587, 0.41004646, 0.58153015, 0.091584764, -0.47686112, 2.703642, 0.7740971, -1.0126826, -1.6270666, 0.14777291, -0.68230283, 2.5803995, -1.7720243, -0.050674938, 0.20010044, 0.14050291, -1.2080921, 0.58900356, 0.30402642, 2.1537213, -0.4592905, 1.3741494, 0.5373596, 0.21713334, 2.137194, 1.2340785, -0.24224578, 2.0349917, -1.930566, -1.3097095, 0.7491717, 0.6829777, -0.75699, 0.5313692, 1.1641297, -0.92273647, 0.4301941, 1.0517148, -1.0875251, 0.6395757, 0.9332728, -0.021947997, 0.17519543, -0.9672251, -0.5985964, 0.04532971, 1.1857414, 0.9184439, 0.35052297, 0.52249724, -0.25172293, -0.30268556, -1.0179735, -0.2219054, 2.3816016, -0.0710321, -0.989576, -0.24906443, -0.65537864, 0.34714538, 0.53644794, -0.92104703, 1.124992, 0.6449487, -0.6661217, 0.33926535, -0.17620465, 0.07624604, 0.72245514, 0.51069385, -1.2656112, -0.1546393, 0.34989673, 1.6776577, -0.14842795, -0.26373982, 0.39450446, -1.9342973, 0.06069861, 0.70746845, -0.06332892, -1.2213986, 0.52144814, -0.0969298, 0.29012161, -2.0257008, -0.6146449, -0.7306237, -0.99169755, -1.7196251, -0.6532671, -0.9064326, -1.9208741, 0.9451501, 0.5264638, 0.8403371, 0.019283814, -1.3982819, -1.8250778, 0.7810097, 1.6849544, -0.6411514, 1.2169693, 0.7479974, 0.2259077, 1.102956, 0.5303258, 0.09757054, -0.56150174, -1.3653239, 0.2555939, -0.11094171, -1.4731196, 1.6103663, -1.5095649, -0.37031603, 0.032324076, -1.2840677, 0.49954963, -1.8749087, 0.56228566, -1.690414, 2.3080883, -2.1366217};
	static CNN_DTYPE CNN_bias[CNN_KERNEL_COUNT] = {0.896784, 0.4265146, 1.1762991, 1.1752667, -3.1464102, 1.0973997, -2.6116688, 1.2558467, 1.5689987, 0.09827021, -0.45260283, 0.5422435, -3.5531642, 1.4501718, 0.603969, 1.3567555, -0.51081944, -2.3065505, 0.56508446, 1.5314894, -2.2898982, -0.06443754, -2.790004, -1.6914426, 1.0447662, 0.75750047, 0.5788959, -0.8906664, 1.1270236, 1.8515958, 0.9702542, -0.9701291};
	static CNN_DTYPE dense_weights[DENSE_INPUT_NODES][DENSE_OUTPUT_NODES] = {0.9278598, 1.1284379, -0.94213384, 0.57213986, -0.15175189, -1.3710651, -0.36759967, 0.23358917, -1.3287755, 1.2052494, 0.29525074, -1.781556, -0.71603924, -0.049652506, 1.1900238, -0.532206, 0.31779057, -1.8949149, 0.33578163, 0.9880877, 2.0349207, 0.24978457, 1.3035396, 1.9510812, -2.221579, -0.9978224, -0.9149573, 0.5963041, -1.3082341, 1.5577532, 1.0436157, 0.090697005, 0.90788054, 2.944364, -2.6875358, -1.1914431, 0.5402517, -0.37729442, -1.2342881, 1.1016369, -1.5059944, -0.5314651, -2.1083255, 1.3691921, 1.1704621, 0.024896603, -2.051138, -0.024866635, 2.6208022, -0.8007732, 1.0956156, 2.2282572, -0.26186803, 1.150465, -1.9988422, 0.1030978, -0.54708993, 1.2227368, -0.63599324, -0.26782066, -3.7935333, -1.7909775, 5.401936, -1.2565079, -3.3145006, 2.2774117, -4.723861, 1.4916931, -3.2078261, -1.6727602, 1.5434126, 4.3297806, 0.59307, -4.884287, -4.0573454, -0.54214776, -2.5495121, -0.19552824, 2.3376946, 0.26311895, -0.15331411, 2.3513055, -1.1189724, 2.9337766, -2.3592186, 1.8632007, 2.413747, 2.139566, -4.228939, -2.6323867, 0.23320854, -0.6490903, -0.6887242, 0.53555113, -0.48436114, -2.1165373, -1.8965633, -0.56331444, 1.5507464, 1.0484781, 3.6116467, 0.57879615, -1.0621713, 1.5341015, -2.076099, 1.7163149, -2.2261753, -1.4174855, -2.3243237, 2.4780533, 2.9355805, 4.629756, 2.023579, 0.03428894, -2.376142, -1.3890179, -0.4306704, 2.2080162, -0.5094435, 0.112367764, 0.09337053, 0.34883454, 0.18785839, -2.152326, 0.6988179, -1.0250405, -1.0070907, -0.11586673, 1.0689838, 1.0615817, -0.61064285, 0.22232966, 1.4815854, -1.1207191, -0.57034653, 0.7056585, -0.20985065, -0.3675305, -2.3084376, 1.3689666, 0.6533726, 1.5941291, -2.5621424, 0.88743573, 0.14838536, -2.1853554, -0.019136315, -0.5558486, 0.9131992, 0.9633786, -1.0049962, -1.6040758, -1.0178531, -0.45557112, 2.0366418, 0.29216012, 0.24305503, 2.805002, -2.0875843, -0.38472292};
	static CNN_DTYPE dense_bias[DENSE_OUTPUT_NODES] = {-0.4578098, -0.18469962, -0.82835203, -0.32413018, 0.48125014};

	static CNN_DTYPE user_0_input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];
	static CNN_DTYPE user_0_cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH];
	static CNN_DTYPE user_0_cnn_output_averaged_buffer[CNN_OUTPUT_DEPTH];
	static CNN_DTYPE user_0_dense_output_buffer[DENSE_OUTPUT_NODES];
	
	static CNN_DTYPE user_1_input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];
	static CNN_DTYPE user_1_cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH];
	static CNN_DTYPE user_1_cnn_output_averaged_buffer[CNN_OUTPUT_DEPTH];
	static CNN_DTYPE user_1_dense_output_buffer[DENSE_OUTPUT_NODES];

	MAIN_PROCESS: if (function_select == 0) {
		if (user_number == 0) {
			// shift input_buffer to the left
			for (int i = 0; i < CNN_KERNEL_LENGTH - 1; i++) {
				for (int j = 0; j < INPUT_DEPTH; j++) {
					user_0_input_buffer[i][j] = user_0_input_buffer[i + 1][j];
				}
			}
			memcpy(user_0_input_buffer[CNN_KERNEL_LENGTH - 1], data, INPUT_DEPTH * sizeof(CNN_DTYPE));

			// evaluate neural network result
			compute_convolution(user_0_input_buffer, CNN_weights, CNN_bias, user_0_cnn_output_buffer);
			compute_global_average_pool(user_0_cnn_output_buffer, user_0_cnn_output_averaged_buffer);
			compute_dense(user_0_cnn_output_averaged_buffer, dense_weights, dense_bias, user_0_dense_output_buffer);
			memcpy(raw_output, user_0_dense_output_buffer, sizeof(float) * DENSE_OUTPUT_NODES);
		} else {
			// shift input_buffer to the left
			for (int i = 0; i < CNN_KERNEL_LENGTH - 1; i++) {
				for (int j = 0; j < INPUT_DEPTH; j++) {
					user_1_input_buffer[i][j] = user_1_input_buffer[i + 1][j];
				}
			}
			memcpy(user_1_input_buffer[CNN_KERNEL_LENGTH - 1], data, INPUT_DEPTH * sizeof(CNN_DTYPE));

			// evaluate neural network result
			compute_convolution(user_1_input_buffer, CNN_weights, CNN_bias, user_1_cnn_output_buffer);
			compute_global_average_pool(user_1_cnn_output_buffer, user_1_cnn_output_averaged_buffer);
			compute_dense(user_1_cnn_output_averaged_buffer, dense_weights, dense_bias, user_1_dense_output_buffer);
			memcpy(raw_output, user_1_dense_output_buffer, sizeof(float) * DENSE_OUTPUT_NODES);
		}

	} else if (function_select == 1) {
		// reset CNN output buffer, CNN output averaged buffer, and dense output buffer
		if (user_number == 0) {
			memset(user_0_input_buffer, 0, sizeof(CNN_DTYPE) * CNN_KERNEL_LENGTH * INPUT_DEPTH);
			memset(user_0_cnn_output_buffer, 0, sizeof(CNN_DTYPE) * CNN_OUTPUT_LENGTH * CNN_OUTPUT_DEPTH);
			memset(user_0_cnn_output_averaged_buffer, 0, sizeof(CNN_DTYPE) * CNN_OUTPUT_DEPTH);
		} else {
			memset(user_1_input_buffer, 0, sizeof(CNN_DTYPE) * CNN_KERNEL_LENGTH * INPUT_DEPTH);
			memset(user_1_cnn_output_buffer, 0, sizeof(CNN_DTYPE) * CNN_OUTPUT_LENGTH * CNN_OUTPUT_DEPTH);
			memset(user_1_cnn_output_averaged_buffer, 0, sizeof(CNN_DTYPE) * CNN_OUTPUT_DEPTH);
		}
		memset(raw_output, 0, sizeof(CNN_DTYPE) * DENSE_OUTPUT_NODES);

	} else if (function_select == 2) {
		// set CNN weights
		memcpy(CNN_weights, weights_and_bias, sizeof(CNN_DTYPE) * CNN_KERNEL_LENGTH*CNN_KERNEL_DEPTH*CNN_KERNEL_COUNT);
	} else if (function_select == 3) {
		// set CNN bias
		memcpy(CNN_bias, weights_and_bias, sizeof(CNN_DTYPE) * CNN_KERNEL_COUNT);
	} else if (function_select == 4) {
		// set dense weights
		memcpy(dense_weights, weights_and_bias, sizeof(DENSE_DTYPE) * DENSE_INPUT_NODES*DENSE_OUTPUT_NODES);
	} else if (function_select == 5) {
		// set dense bias
		memcpy(dense_bias, weights_and_bias, sizeof(DENSE_DTYPE) * DENSE_OUTPUT_NODES);
	} else {
		// do nothing
	}
}
