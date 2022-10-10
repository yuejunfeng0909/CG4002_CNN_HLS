#include "cnn_model.h"
#include <string.h>

void cnn_action_detection(
		FUNCTION_SELECT_BIT_WIDTH function_select,
		float data_in[CNN_KERNEL_LENGTH*INPUT_DEPTH],
		int &result_out,
		int &data_required,
		float raw_output[DENSE_OUTPUT_NODES],
		float cnn_average_output[CNN_OUTPUT_DEPTH],
		float cnn_output[CNN_OUTPUT_LENGTH*CNN_OUTPUT_DEPTH],
		float weights_and_bias[CNN_KERNEL_COUNT * CNN_KERNEL_LENGTH * CNN_KERNEL_DEPTH]) {
#pragma HLS INTERFACE mode=s_axilite port=function_select
#pragma HLS INTERFACE mode=m_axi port=data_in depth=90 offset=slave
#pragma HLS INTERFACE mode=s_axilite port=result_out
#pragma HLS INTERFACE mode=s_axilite port=data_required
#pragma HLS INTERFACE mode=m_axi port=raw_output depth=3 offset=slave
#pragma HLS INTERFACE mode=m_axi port=cnn_output depth=130 offset=slave
#pragma HLS INTERFACE mode=m_axi port=weights_and_bias depth=900 offset=slave
#pragma HLS interface s_axilite port=return

	// Initialize weights and bias
	static CNN_DTYPE CNN_weights[CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH][CNN_KERNEL_COUNT] = {-0.019732883, 1.8484143, -0.39383218, 2.0958884, 0.8869749, 0.71870315, -0.3923508, 0.12545647, 0.20025367, -0.17483832, 1.0230312, -0.055513978, 0.12743874, 1.0870754, 1.6485134, 1.0899471, -0.2343243, -0.22319625, 0.26656923, -0.4711292, -0.0394503, -0.39481267, 1.109989, -0.028537624, -0.32806355, 0.57035863, 0.27115414, 0.9734824, 0.08500467, 1.1345934, 1.4569837, -0.19617008, -0.3019426, 0.90139276, 0.51037014, -0.39057106, 1.3119115, -0.27525207, 1.3428866, -0.116335645, -0.13149065, 0.059442867, -0.7805228, 0.30411515, -0.21808954, 2.438467, 0.52748, -0.486242, 0.397193, 0.718725, -0.71457165, 0.1619382, -0.17017606, 0.38184485, 0.40912083, 0.0132769365, 0.12631573, -0.7440546, -0.011844432, 2.0990095, 0.033201907, 0.09179741, 0.71773124, 0.19912322, 0.17301941, -0.15728085, 0.73660165, 0.6285052, 0.58470345, -0.80596423, 0.5832864, -0.12589724, -1.0170096, 1.125905, 0.736691, -0.082744054, 0.92587644, 0.20282139, -1.2152296, 0.8455644, -0.36163992, -0.06536492, -0.45063916, 1.0632885, 0.23056507, 0.5818826, -0.1256059, 0.8599611, 1.0925715, -0.17603098, -0.40173748, 0.6449782, -0.6146549, -0.35764948, -0.28342563, 0.9486916, 0.30952308, -0.25806174, -0.10998334, -0.596963, -0.45342338, 0.3749626, 0.06694922, 0.10738586, 0.25431734, -0.5503733, -0.3605465, 0.27777544, 0.06400271, -0.10386786, 0.087655164, -0.46549475, -0.090207435, -0.6029549, 0.04829311, 0.090634786, -0.0059883893, 0.56355184, -0.3917573, -0.7516828, -0.7096759, 0.21801609, 0.32128537, 0.12262767, -0.07213552, 0.21414761, -0.23876314, 0.23066926, 0.4389052, -0.22672595, 0.43681258, 0.030986656, 0.29216307, 0.30716148, -0.010958332, -0.5816872, 0.34103486, 0.4785414, -0.78057116, 0.07606966, -0.29450482, -0.51461756, 0.099570684, 0.1093335, 0.17297004, -0.59304494, 0.22083974, -0.033650395, 0.064834915, 0.48329532, 0.30357763, 0.07566626, 0.031143516, -0.33122405, 0.049059402, -0.21781272, 0.11291629, -0.88540137, -0.3579574, -0.20652704, 0.19609198, 0.8946683, -0.25506163, 0.8686231, 0.8970837, 0.2981903, -0.021948986, -0.2787653, -0.010906051, 0.37599024, -0.5060322, -0.13453792, 0.9428085, 0.5069109, 0.523877, 0.66731113, 0.5330404, -0.24859108, 0.32607186, -0.32624963, -0.4017699, -0.70068026, 0.15725645, 0.34857547, -0.1271144, -0.40434045, 0.0932039, 0.12283421, -0.26149264, 0.5750368, 1.0226399, 0.14678933, -0.02629766, 2.06085, -0.4666055, 2.049076, 0.63517314, 0.8055182, -0.34145233, 0.058183793, 0.3535947, -0.062322788, 0.95617986, -0.25682795, -0.2961848, 0.6661382, 1.5080678, 0.9716408, -0.3772722, -0.15024415, 0.12325131, -0.20847863, -0.16843387, -0.38181406, 1.0684721, 0.34560975, -0.34522408, 0.8415944, -0.07998889, 0.86693424, -0.1750385, 1.2454973, 1.7986559, -0.2434245, -0.45657942, 0.9783979, 0.46096757, -0.3740883, 1.2331002, -0.07846186, 0.75085026, -0.15734987, -0.21083218, 0.3224089, -0.43127674, 0.23724595, 0.15283325, 1.9773232, 0.25050512, -0.43614402, 0.375007, 0.36961278, -0.6156249, 0.13118039, 0.18760781, 0.43359452, -0.13705233, 0.1101965, -0.021921992, -0.5375031, 0.3336359, 1.8931491, 0.04583246, 0.2710942, 1.0984266, 0.31845298, 0.21242288, -0.14262782, 0.6141104, 0.4352803, 0.54628557, -0.9041283, 0.82682836, -0.13278514, -0.78227764, 1.1656348, 1.0670087, 0.1386791, 0.90466684, -0.5526552, -1.4817324, 0.6573062, -0.20703407, 0.30941138, -0.14865918, 0.90886295, -0.21626149, 0.22834012, -0.6320482, 0.6345924, 1.3222225, -0.22254086, -0.18121827, -0.08560408, -0.47425893, 0.1643999, 0.012544649, 1.0655227, -0.07171278, -0.1757902, 0.10150475, 0.037891723, 0.061162136, -0.15723513, 0.17909932, -0.024183383, 0.009769775, 0.07606581, -0.18340296, 0.12618548, 0.07257027, 0.12409848, 0.124329306, 0.21782875, 0.12535271, -0.044641584, 0.15287113, 0.1124558, 0.27196768, -0.13871178, -0.087297834, -0.3291227, 0.03656379, 0.64592415, 0.35663155, 0.1090363, 0.15324347, 0.081313334, -0.1890761, 0.27268443, -0.22318666, -0.12585391, 0.23324127, 0.2343263, 0.10467883, -0.015568021, -0.021132072, 0.13534169, 0.24984519, 0.06381373, -0.3889698, 0.039083365, -0.11733788, -0.34859616, -0.23250799, 0.16964781, 0.1088383, -0.3612114, 0.29908693, 0.045097608, 0.17011599, 0.10022616, 0.16144454, 0.14710303, 0.13038166, -0.014073414, 0.0496899, -1.0411216, -0.10740972, -0.5356306, -0.17171937, -0.08535902, -0.26054475, 0.45536342, -0.49955416, 0.33290374, 0.5235561, -0.0996048, -0.33885366, 0.12993304, -0.05580111, 0.022647962, -0.011395737, -0.17279865, 0.27006498, -0.047515806, 0.1504025, 0.23896332, -0.13109711, -0.13713571, 0.37566563, -0.036994915, 0.0003079526, -0.0032047112, 0.23168156, 0.3681934, 0.07416845, -0.032500155, 0.12147648, -0.15959144, -0.24441274, 0.40255657, 0.19610025, 0.3378528, 0.013317692, 1.8483508, -0.053333696, 2.2277098, 0.343168, 1.2069751, -0.6465763, 0.31801665, 0.49022865, -0.112836145, 0.8976861, -0.21299265, -0.006134085, -0.054367565, 1.1905388, 0.66035193, -0.08450332, -0.25469476, 0.11195782, -0.31289765, -0.32417032, -0.25458032, 0.6840114, 0.46583462, -0.5637422, 1.0227748, -0.09036892, 0.7627471, -0.1518235, 1.2280933, 1.467218, -0.5779148, -0.3979384, 1.2232512, -0.36567935, -0.7129897, 1.6191982, -0.1346785, 0.6800778, -0.42662147, 0.16721071, 0.48268813, -0.41203976, 0.4520128, -0.080027014, 2.2418456, 0.2756389, -0.7471228, 0.24501997, 0.36258894, -0.46811545, 0.22373325, 0.34763494, 0.38701534, -0.22846356, 0.36884055, 0.120104775, -0.45435566, 0.11915333, 2.2358289, -0.28891134, 0.4401009, 0.9867312, -0.022955239, 0.5429152, 0.11901214, 0.6497085, 0.3536136, 1.0673044, -0.64882916, 0.8674421, -0.014748374, -0.5460493, 0.8359278, 1.064157, -0.20691285, 1.3620337, -1.0620121, -1.489136, 0.152325, -0.589562, 0.5431616, -0.2342748, 0.8103551, -0.1501888, 0.19717328, -0.73085177, 0.97969306, 0.9393423, -0.19639619, 0.015263106, -0.6676494, -0.4156308, 0.4038973, 0.24521212, 0.6396014, -0.5418502, -0.43492672, -0.0018434762, -0.00862352, -0.16416979, -0.38076937, 0.26384252, -0.29887652, -0.61522645, 0.311661, 0.083263785, -0.03257641, -0.1847521, 0.6366749, 0.42726445, 0.14265439, -0.29648423, 0.64583015, 0.5957475, -0.20095176, 0.23213562, -0.36488423, -0.6638286, -0.041428618, 0.27114078, 0.1003087, 0.16658925, 0.54141533, 0.6143665, 0.039793145, -0.2424988, -0.00780228, -0.5014702, -0.1696787, -0.13403887, 0.16159523, -0.13393149, -0.42943966, -0.06546818, 0.47576314, 0.058264554, 0.330794, -0.15303005, 0.32325646, -0.19976643, -0.4009101, -0.101228476, 0.4936634, 0.37362862, -0.16447783, 0.10803146, -0.0010481643, 0.04972807, 0.07657335, 0.47584915, 0.5198191, 0.36083758, 0.4590004, -0.5757877, -0.41581926, -0.103801414, -0.09035959, 0.07950852, -0.009413325, -0.6161811, -0.112254575, -0.5053152, -0.20295028, 0.6152057, -0.8849005, -0.671234, 0.38307688, -0.49487457, 0.01577845, -0.6130547, -0.010529412, -0.35109538, -0.1530591, 0.45405233, -0.6387916, -0.7366312, -0.2346559, -0.124357805, -0.03165834, 0.044324636, 0.61416745, -0.3642299, -0.034637515, -0.23397899, 0.16365972, 0.1651011, -0.12593304, 0.14513715, 0.23388676, -0.186029, -0.058560677};
	static CNN_DTYPE CNN_bias[CNN_KERNEL_COUNT] = {0.22052337, -0.65572214, -0.630265, -0.25072518, -0.8846231, -0.94624794, -1.5546609, 0.24620414, -1.114795, -1.1195322, -0.91270626, 0.1208914, -0.526202, -2.3044183, -1.8499374, -1.2915399, -0.5266661, -0.07485161, -0.16327918, 0.18583411, -0.03661107, -0.08441811, -0.60984504, -0.79557186, -1.1828408, 0.07544267, 0.1562624, -2.4836571, -0.16530703, 0.7381609, -0.51033974, 0.046812188};
	static CNN_DTYPE dense_weights[DENSE_INPUT_NODES][DENSE_OUTPUT_NODES] = {0.34469026, 0.66614, -0.116819315, -0.12438927, 1.0407584, -2.571108, -0.36173907, -2.0994523, 0.20376877, -0.58932763, 0.9064206, -0.8838202, 0.44286966, -1.9535853, 0.88831466, -2.5371807, 0.34555942, 0.90298563, -1.4295728, 1.882473, 0.96715164, 1.5182999, -1.9524662, -0.2578153, -0.26332775, 0.76931953, 1.1652128, -2.612621, -0.4958865, 0.44109213, 0.29968026, -0.5156511, 0.68922, 0.37163734, -1.8357346, 0.98265713, -1.4024044, -2.0559494, -0.036155473, 1.0851346, -1.1566488, -0.08554345, -0.010687428, 1.1230844, 0.27345863, 0.41814637, -0.5059949, 0.34765315, 0.97935075, 0.20795861, -0.5320265, 0.5124743, -1.9431835, -1.1726631, 2.486696, -0.8152642, -2.303552, 1.8211057, 0.065025724, -0.6964357, -0.66298294, 0.48926544, 1.7426854, -2.6159859, 0.65928227, 1.0458577, -0.33058408, -0.8445883, 0.12895101, -1.0331844, 0.30619982, 0.80758584, 0.0049611623, -0.12390059, 0.6024295, -0.31178227, -0.7919954, 0.5709176, 0.1027086, 0.7531282, 0.29439798, -0.2554675, -0.28086734, 0.051887505, -1.1133013, 0.13084613, 0.07018191, 0.8454862, -0.6741294, 1.5973492, -0.2496403, -1.3414598, -0.2640419, -1.7043874, 0.091462225, -0.22013764, -0.9177387, -2.074621, 0.82105297, 0.69784063, 0.37304366, -0.22778088, -0.7437912, 1.2509798, -0.44631112, 0.20678051, 0.25304326, -0.42453262, -0.6902733, -0.79701173, 2.4739137, -3.398496, 0.21657275, -0.26691535, 0.30649754, 0.07044383, 0.28551754, 1.3119389, -0.18361275, -1.632223, 1.3228949, -2.2870414, -0.3565169, -2.641622, 0.93225056, 0.45585397, -0.8821896, 0.5144007};
	static CNN_DTYPE dense_bias[DENSE_OUTPUT_NODES] = {0.03936374, -0.021348884, 0.08467927, -0.10721366};

	static CNN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];
	static CNN_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH];
	static CNN_DTYPE cnn_output_averaged_buffer[CNN_OUTPUT_DEPTH];
	static DENSE_DTYPE dense_output_buffer[DENSE_OUTPUT_NODES];

	static int CNN_output_free = CNN_OUTPUT_LENGTH;

	MAIN_PROCESS: if (function_select == 0) {

		// input more data
		memcpy(input_buffer, data_in, sizeof(CNN_DTYPE) * CNN_KERNEL_LENGTH*INPUT_DEPTH);
		compute_convolution(input_buffer, CNN_weights, CNN_bias, cnn_output_buffer);
		memcpy(cnn_output, cnn_output_buffer, sizeof(CNN_DTYPE) * CNN_OUTPUT_LENGTH*CNN_OUTPUT_DEPTH);
		compute_global_average_pool(cnn_output_buffer, cnn_output_averaged_buffer);
		memcpy(cnn_average_output, cnn_output_averaged_buffer, sizeof(CNN_DTYPE) * CNN_OUTPUT_DEPTH);

		// compute dense, need software check if result is valid (data_required == 0)
		compute_dense(cnn_output_averaged_buffer, dense_weights, dense_bias, dense_output_buffer);
		memcpy(raw_output, dense_output_buffer, sizeof(DENSE_DTYPE) * DENSE_OUTPUT_NODES);
		argmax(dense_output_buffer, result_out);
		CNN_output_free = (CNN_output_free == 0) ? 0 : CNN_output_free - 1;
		data_required = CNN_output_free;

	} else if (function_select == 1) {

		// (RESERVED DEBUG FUNCTION)

	} else if (function_select == 2) {

		// reset CNN output buffer
		memset(cnn_output_buffer, 0, sizeof(CNN_DTYPE) * CNN_OUTPUT_LENGTH*CNN_OUTPUT_DEPTH);

		// reset raw output buffer
		memset(raw_output, 0, sizeof(CNN_DTYPE) * DENSE_OUTPUT_NODES);

		// reset the number of data required
		CNN_output_free = CNN_OUTPUT_LENGTH;
		data_required = CNN_output_free;

	} else if (function_select == 3) {
		// set CNN weights
		memcpy(CNN_weights, weights_and_bias, sizeof(CNN_DTYPE) * CNN_KERNEL_LENGTH*CNN_KERNEL_DEPTH*CNN_KERNEL_COUNT);
	} else if (function_select == 4) {
		// set CNN bias
		memcpy(CNN_bias, weights_and_bias, sizeof(CNN_DTYPE) * CNN_KERNEL_COUNT);
	} else if (function_select == 5) {
		// set dense weights
		memcpy(dense_weights, weights_and_bias, sizeof(DENSE_DTYPE) * DENSE_INPUT_NODES*DENSE_OUTPUT_NODES);
	} else if (function_select == 6) {
		// set dense bias
		memcpy(dense_bias, weights_and_bias, sizeof(DENSE_DTYPE) * DENSE_OUTPUT_NODES);
	} else {
		// do nothing
	}
}
