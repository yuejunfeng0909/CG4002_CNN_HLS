#include "cnn_model.h"
#include <string.h>
#include "set_weight_bias.h"

// set up stream
// #include "ap_int.h"
//#include "hls_stream.h"
//
// typedef ap_int<16> int16_u;  // 16-bit raw data width
//hls::stream<uint16_t> data_in_stream;  // A stream declaration

// int16_u something;

void cnn_action_detection(
		FUNCTION_SELECT_BIT_WIDTH function_select,
		CNN_RAW_IN_DTYPE data_in[CNN_KERNEL_LENGTH*INPUT_DEPTH],
		int &result_out,
		int &data_required,
		float raw_output[DENSE_OUTPUT_NODES],
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

	static INPUT_DTYPE CNN_weights[CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH][CNN_KERNEL_COUNT] = {-0.042539876, 0.08948778, 0.0853378, -0.16634205, 0.12144841, -0.025482206, -0.038470566, 0.18711008, 0.12927802, 0.05440186, -0.100298636, -0.11426955, -0.14722964, -0.0027086432, 0.053133257, -0.026186038, 0.008446946, -0.08776002, -0.04001655, -0.029527025, 0.04844884, -0.14158957, -0.14926258, 0.14551324, -0.0051613715, 0.011370315, 0.12194996, 0.10195985, -0.014585853, 0.04128646, 0.08046318, 0.0482731, -0.026146732, -0.15721686, 0.039504502, 0.17034134, 0.11296605, 0.077609435, 0.1810051, -0.12484771, 0.15639962, 0.062023193, -0.01974073, -0.12848325, 0.15445165, -0.061539274, 0.06454646, 0.13184573, 0.013708684, 0.1414473, -0.04438883, -0.057954468, 0.12254742, -0.056726765, 0.12271303, -0.114431225, 0.05082047, -0.079468146, -0.03768174, 0.031847578, 0.118551165, -0.042846054, -0.050900433, -0.10780252, 0.11338932, 0.11775639, 0.026648799, 0.0868099, -0.077264905, -0.037421174, 0.11233048, 0.04180971, -0.14320025, -0.023912035, 0.092102036, 0.11796927, 0.15283711, 0.030390484, -0.090355895, -0.061064318, -0.07827843, -0.072294295, 0.006047205, -0.08086747, 0.009775724, 0.12830733, 0.16291143, 0.18516108, -0.11039302, 0.09197364, 0.08260048, 0.12374439, 0.11403559, -0.12658708, -0.08664876, 0.111895666, 0.07688376, 0.009053663, -0.051730387, 0.11247332, 0.1511784, 0.028809264, 0.028009335, -0.030841963, 0.035000615, -0.16345868, 0.021548998, 0.046156254, -0.05706776, 0.15597896, -0.11315426, -0.086719, -0.07388202, 0.11138578, 0.06838601, -0.0019759692, -0.14390887, 0.10512755, -0.06964548, -0.06996794, 0.081036456, -0.10816075, -0.13424681, -0.03369487, -0.10200494, -0.013996202, 0.053556297, 0.023570728, -0.104961574, -0.09862344, 0.12917748, -0.10424309, -0.058997676, 0.13654335, 0.010960547, -0.14126855, -0.10572729, 0.13732617, -0.098046236, -0.07947249, 0.09606285, -0.0790237, -0.20577857, 0.115075916, 0.1304689, 0.14337651, 0.1744947, 0.17821397, 0.14497437, -0.14748473, 0.038052443, 0.074687906, 0.068944834, -0.1729719, -0.16429114, 0.07297632, -0.049747713, -0.11265068, -0.023136018, -0.13873261, 0.056304935, -0.11125476, 0.088784546, 0.1601584, -0.027140327, -0.038825713, -0.07029959, -0.018786276, 0.07367793, 0.11845138, -0.03303995, 0.03171017, -0.060153704, -0.02657127, 0.13715601, -0.029134143, 0.053225122, 0.07857967, 0.11873735, -0.03647922, 0.13801892, 0.10438976, -0.052421033, 0.1120447, -0.06005838, -0.04789715, -0.11564658, 0.11903691, 0.08828388, 0.042862177, -0.10886549, 0.0136223715, 0.051205322, 0.020768955, -0.04808691, 0.1011861, -0.052800745, 0.08466451, -0.07370095, -0.047222905, 0.017131455, -0.18411328, -0.08022317, 0.04789521, -0.08152143, -0.034993447, -0.0133596705, 0.04917978, 0.046121307, -0.1373841, 0.04228371, -0.1618745, -0.18287078, -0.05577391, 0.07561807, -0.12359046, 0.15096071, 0.013207972, -0.0015910813, -0.027180374, -0.12879986, 0.14030315, 0.035208564, -0.063925244, 0.08901283, 0.0817166, 0.13795657, 0.12852192, 0.04272112, -0.086373635, 0.002558458, -0.032981813, -0.064673826, 0.018966554, 0.055100124, -0.16519505, 0.008426132, 0.080535874, -0.07695402, -0.064528465, -0.027637625, 0.05744182, -0.090969644, -0.036547948, 0.119255275, 0.008743683, 0.0075773722, 0.036027025, -0.15491255, 0.113106385, -0.08930217, -0.06468744, -0.098788254, -0.14803109, 0.11168038, -0.044999413, 0.106456645, -0.10174789, -0.044195943, -0.026982347, -0.07771326, -0.03110685, -0.024749314, -0.11868663, 0.1311215, 0.0022220737, -0.021949159, 0.18603985, -0.0032831926, -0.042040862, 0.1285295, -0.104122646, 0.04542268, -0.16934134, -0.015363922, 0.1689467, -0.05476088, -0.13389203, -0.02825149, 0.056481175, 0.09983993, 0.03971241, 0.043360807, -0.0047333604, 0.12136801, 0.05520062, -7.347289e-05, 0.010267805, 0.13643633, 0.019079605, 0.02998717, -0.06802152, 0.0015713813, 0.06966924, -0.040266033, -0.13258632, -0.074911125, 0.055006064, -0.053894274, -0.03318353, 0.061103333, -0.014834727, -0.04737752, -0.14812732, 0.054082, -0.07936893, 0.07265848, 0.08420115, 0.097298585, 0.14992195, 0.060919058, -0.13278809, -0.1756479, 0.014674134, 0.04091363, -0.084380165, -0.025938876, 0.14924555, 0.13981338, 0.067883454, 0.15127695, -0.08135089, -0.060079012, -0.08279259, 0.059373308, 0.18771337, 0.18864554, -0.037873875, 0.0132970335, 0.0018257995, -0.114599735, -0.16501758, 0.015282575, 3.923577e-05, 0.09951781, 0.003856832, -0.14260577, 0.026788965, -0.10139175, -0.12969002, -0.123868585, -0.086858496, -0.11267923, -0.113001265, 0.051141236, -0.049099937, -0.07494292, 0.087612376, 0.031564936, -0.11918152, 0.11532289, -0.031384572, -0.0069112848, 0.08959687, 0.15808104, -0.17626746, 0.08751772, -0.019124897, 0.08371432, 0.09195196, 0.11000922, -0.11172825, 0.108166955, 0.005469834, -0.065165974, 0.110286966, -0.09699833, 0.006629664, -0.060822777, 0.1471742, 0.074444644, 0.103833064, -0.16434015, -0.17638622, 0.081727184, 0.037751123, 0.044270165, 0.016488766, 0.069415845, -0.12664409, -0.05079089, -0.10354446, -0.06761607, -0.10895797, 0.033240777, 0.20885786, -0.062259313, -0.062917404, -0.0522747, -0.008326976, 0.16598515, -0.03973113, 0.072133385, -0.01520275, 0.118431605, 0.07295946, -0.09672971, 0.13878892, 0.0035710402, -0.08639139, 0.07168477, 0.054157652, 0.1338708, 0.13825558, -0.06530282, -0.029201416, 0.07279959, 0.09853258, 0.09790309, 0.106449224, 0.10649328, 0.11743789, 0.05399516, -0.18705145, 0.06669209, -0.0006968953, 0.053343575, 0.0010580466, -0.060677573, -0.08399476, 0.10673758, -0.005044184, 0.033696122, 0.0241195, 0.13125837, 0.17003839, 0.15507676, 0.13954946, -0.10314067, 0.07220564, -0.082249366, -0.06131445, -0.17936167, 0.045129057, 0.0973142, 0.12977998, -0.081027865, 0.056258682, -0.12215969, -0.12807332, 0.18627436, -0.010645892, -0.15285212, 0.13946483, 0.083191946, 0.16967072, -0.05473435, -0.013331026, -0.08578795, 0.04640058, -0.08146179, 0.12557687, 0.0023099135, 0.012543855, 0.09116714, 0.0034418327, -0.09099241, 0.0062593855, 0.18885449, -0.034372214, 0.1637309, -0.01222662, -0.06102142, 0.13710359, 0.007431563, -0.00071350514, 0.008334358, 0.08409914, 0.053069744, 0.093602665, -0.1643586, 0.093022086, 0.04193692, 0.097659625, 0.0025582912, 0.03714544, -0.11480746, 0.13716386, -0.0789411, 0.060203675, -0.10664833, 0.09670777, 0.15290791, -0.14552896, -0.11364957, 0.12277719, 0.14799766, -0.06379067, 0.017007243, -0.09435814, -0.103877306, -0.01089741, 0.0081581725, -0.07907345, -0.065696605, 0.015193498, -0.026662773, -0.038870394, 0.15085635, -0.03303259, 0.16846332, 0.042755447, 0.03223733, 0.022046367, 0.051816545, 0.10984414, -0.087414704, 0.06848547, -0.064037316, 0.049168028, -0.1047204, 0.065923594, 0.09440381, -0.1582679, -0.09253758, -0.0663419, 0.008432192, 0.029061936, 0.18373339, -0.11621416, 0.06799474, 0.11064179, -0.04968458, -0.06803283, 0.09546297, 0.043992665, 0.13269977, -0.0464847, -0.10957779, -0.08945254, -0.0855057, 0.03524574, 0.030066725, -0.1467524, 0.07977687, -0.1455552, 0.017293729, 0.19032213, -0.18604976, -0.14447372, 0.12288018, 0.12559655, 0.13004312, 0.113780946, 0.1657233, 0.15046942, -0.11985374, -0.10828654, 0.076389104, -0.061899174, 0.09388358, -0.08875922, 0.08909476, -0.043915942, 0.119166054, -0.14902571, 0.10405414, -0.09911105, -0.1079436, 0.10193246, 0.034783427, -0.17687961, 0.08122158, 0.012883848, -0.009738098, 0.024566595, 0.09837345, 0.1363063, -0.1191098, -0.11439973, -0.03558525, -0.0013853977, -0.15039133, -0.050851695, 0.001743385, -0.09377184, -0.10286916, -0.0473214, -0.11505093, 0.10624581, -0.094738334, 0.09506416, -0.09388999, -0.07626362, 0.14843087, 0.019676402, 0.1270193, 0.10212458, 0.12404154, 0.072582185, -0.03412555, 0.056285217, 0.07355517, 0.025083048, 0.074081674, -0.12450335, 0.0085326135, 0.04572947, -0.00048524025, 0.09083063, -0.063464485, -0.003043698, -0.12707083, -0.033335365, -0.074948244, 0.016213704, 0.07827475, 0.14213105, -0.011871902, -0.11462313, -0.13387445, 0.04880885, -0.10063748, -0.08289069, -0.047989998, -0.15591997, 0.014926594, 0.06505367, 0.13019581, -0.12724218, -0.11999202, -0.003988038, -0.15318844, 0.12028909, -0.02049361, 0.10679562, -0.07896241, 0.16035296, -0.07294098, -0.17414226, -0.09455691, 0.008824461, -0.052257873, -0.049287274, 0.12956463, -0.095703274, -0.057398986, 0.0041627684, 0.13859108, -0.08454049, -0.1349033, 0.046171483, 0.048552994, 0.008666712, -0.11870013, 0.033778097, -0.14851707, 0.11117902, 0.15437326, -0.078970835, 0.085762076, 0.1337359, -0.10680407, -0.04821599, 0.08354666, -0.006211745, -0.08649981, -0.067488305, -0.1807462, -0.09327608, -0.0120635675, 0.11848255, 0.13079546, -0.032541152, 0.16953063, 0.17307532, 0.18594472, 0.059964575, 0.06459561, -0.0056412965, 0.100238726, -0.16469383, -0.110710844, -0.14188835, -0.046577867, 0.12945203, -0.09868342, -0.086322725, -0.10674578, -0.091874264, -0.055352423, -0.19774577, -0.11113633, 0.071729295, -0.05191341, -0.063827686, 0.11883343, 0.15721081, 0.088508606, -0.1224836, -0.074657656, -0.05823847, -0.08785535, -0.17579478, -0.14093786, 0.010048955, 0.09947528, -0.009896663, 0.035643198, -0.08523035, -0.124784574, -0.027955556, -0.103175364, -0.11418625, 0.019503042, -0.13075255, -0.060067102, 0.08050062, -0.12652351, 0.049490437, -0.151168, 0.12485356, 0.06148383, -0.053682458, 0.041257195, 0.12336071, 0.030784832, 0.118321106, 0.018693892, -0.011755683, 0.03646234, 0.12383803, 0.046810787, -0.11231686, 0.10995514, 0.008332794, -0.09359123, -0.015482693, 0.012897829, 0.02336726, 0.01271885, -0.028426236, 0.09529699, -0.02161979, 0.10998822, -0.065214925, -0.026861558, 0.054343045, -0.028572267, -0.044470824, 0.14993402, -0.047385253, -0.1352959, 0.06747338, -0.01657648, 0.041475646, 0.09531829, 0.14354138, 0.10681139, -0.15937667, -0.039963428, 0.031676095, -0.09577046, 0.085767895, -0.041703418, -0.08311555, -0.025098471, -0.048231248, 0.06929812, 0.031845957, -0.08114231, -0.010925584, 0.12580027, 0.15126696, 0.15524837, -0.17043062, -0.001337706, -0.06537058, -0.09679004, 0.07659341, 0.0714058, 0.008248606, 0.025357783, 0.03331366, -0.084977, 0.1426849, 0.027403902, 0.1841924, 0.097972, -0.14940841, -0.12604167, -0.0559228, 0.1012123, 0.051012985, -0.11857998, 0.13529658, -0.068639815, -0.123849265, 0.0720515, 0.13056153, 0.02226475, 0.10452335, -0.0016292337, 0.009105918, -0.09468336, 0.10490858, -0.04773017, 0.14600535, 0.052960373, 0.09023916, -0.059752364, 0.087639354, 0.04706322, 0.08839438, -0.06750236, 0.0028141993, 0.025434222, -0.05991157, 0.087703876, -0.112072565, 0.11362822, -0.0067123137, -0.1400647, 0.09839795, -0.13077688, -0.020893598, 0.12116929, 0.11335397, -0.042853188, 0.1544391, -0.033100095, 0.0024269286, 0.031636674, 0.17352965, 0.09959984, -0.094073586, -0.042136304, 0.0667448, 0.052778978, 0.14279203, 0.13594677, 0.089151345, 0.09066328, 0.099589154, -0.069631755, -0.004035148, 0.052926153, -0.040813178, 0.11071318, 0.013500047, 0.107377775, -0.025155928, -0.111239016, -0.13684335, 0.080738164, 0.18003528, 0.1484808, 0.07412594, -0.1311789, -0.030659597, -0.10482851, 0.13253486, -0.15163007, -0.02730618, 0.025227377, 0.12562905, 0.096082196, -0.14414759, -0.0036395506, -0.093157396, 0.05677426, -0.05547511, 0.026114438, 0.07567283, 0.096372284, 0.0119290035, -0.08217631, 0.15156253, 0.048363242, -0.06290628, 0.10652055, -0.14648849, -0.1358369, -0.08415522, -0.1464661, 0.060900684, 0.024152575, -0.05030999, 0.106695876, 0.10556442, 0.09657544, -0.032205854, -0.07041581, -0.07700079, 0.16334468, -0.15307532, 0.056527395, 0.1255944, -0.12404404, 0.04999324, 0.0799991, 0.0061892904, 0.08779127, -0.103767976, -0.08120904, 0.03383906, -0.11900252, -0.055908263, -0.10178056, 0.068557024};
	static INPUT_DTYPE CNN_bias[CNN_KERNEL_COUNT] = {0.00011849993, -0.022435242, 0.03585649, 0.0051824604, -0.026486173, 0.04025192, 0.016067352, 0.0055243555, -0.008814534, 0.026299264};
	static INPUT_DTYPE dense_weights[DENSE_INPUT_NODES][DENSE_OUTPUT_NODES] = {-0.1171829, 0.13691343, 0.09600926, -0.2738003, 0.17552489, -0.09069515, -0.13678269, 0.25994608, 0.14640838, -0.037155744, -0.08849762, -0.19093727, -0.12174974, -0.0036653616, 0.0094217425, -0.04904952, -0.0388057, -0.10540566, -0.06360902, 0.013467839, 0.016273052, -0.16803579, -0.14870004, 0.23519209, -0.04111989, -0.059830066, 0.11242018, 0.050447177, 0.003786332, 0.06943764, 0.04082008, 0.11205645, 0.030929947, -0.22517367, 0.124402024, 0.14762187, 0.08542785, 0.14691207, 0.13507651, -0.26234865, 0.22909307, 0.03801467, -0.052967183, -0.20651543, 0.15908606, -0.118411094, 0.08332615, 0.2447142, 0.054469895, 0.12588908, -0.05217682, -0.18269525, 0.06284499, 0.009701355, 0.18119022, -0.14367475, 0.041487236, -0.11545237, 0.015717193, -0.016122283, 0.10222865, -0.07321034, -0.05670808, -0.17748271, 0.16084294, 0.09283125, -0.043845475, 0.12506153, -0.11408643, -0.13784173, 0.15280882, 0.07471678, -0.13801663, 0.0050622374, 0.06347737, 0.13576555, 0.16338494, 0.08343094, -0.099555746, -0.059128992, -0.16343606, -0.079505876, 0.05100858, -0.06294634, 0.03796872, 0.06844099, 0.13177693, 0.18386285, -0.12282188, 0.12461099, 0.054380298, 0.21473992, 0.21716848, -0.17493488, -0.060593206, 0.09929106, 0.03239417, 0.051980224, -0.17013454, 0.059463557, 0.22550504, 0.034244243, -0.034197222, -0.066283815, 0.009030802, -0.20015125, 0.028679842, 0.09796685, -0.054856285, 0.14141242, -0.13009949, -0.19918555, -0.17912324, 0.2293845, 0.11202637, 0.0141487755, -0.24238704, 0.10762201, -0.032260455, -0.14039247, 0.053853437, -0.11216481, -0.20616993, -0.07718134, -0.13703425, -0.071765825, 0.006293222, 0.04755154, -0.16354764, -0.23392262, 0.16993573, -0.10375486, -0.05537929, 0.2065458, -0.020735988, -0.13135816, -0.19672704, 0.1684382, -0.13315207, -0.04450881, 0.06839258, -0.070503116, -0.22818245, 0.2048111, 0.17355031, 0.10982996, 0.14839198, 0.18791197, 0.19473438, -0.18606798, -0.034349583, 0.12598222, 0.1575355, -0.23262759, -0.1547002, 0.030199116, -0.11295557, -0.11724872, -0.12489538, -0.2742123, 0.0941689, -0.1367064, 0.03786376, 0.1862423, -0.056647703, -0.004629303, -0.13357408, 0.004148822, 0.115903765, 0.10701836, -0.032351453, -0.03768483, -0.14583528, 0.04929032, 0.1828661, -0.018861404, 0.03188151, 0.05288397, 0.22416982, -0.08438802, 0.120257065, 0.1829699, -0.08801016, 0.11143413, -0.0643929, -0.1250484, -0.1716552, 0.17003934, 0.08287637, -0.049769357, -0.10611328, -0.0027477038, 0.07379607, 0.06298644, -0.08738248, 0.1640784, -0.14620087, 0.13252452, -0.09845226, -0.013397688, -0.032358088, -0.21283898, -0.054782882, 0.113224626, -0.12571226, -0.1267259, -0.10501398, 0.0044959905, 0.063657254, -0.16230504, -0.013611365, -0.18687421, -0.17750387, -0.09027606, 0.15434334, -0.17404538, 0.14447169, 0.060656283, -0.09442841, -0.1241976, -0.15122654, 0.1866987, -0.028753152, -0.10060958, 0.09124663, 0.1937233, 0.14596559, 0.18731095, 0.07386537, -0.17417493, 0.011452262, -0.1301018, -0.16122544, 0.12296452, 0.066190526, -0.18696018, -0.033499118, 0.022889396, -0.011616438, -0.13134076, -0.104487136, 0.11210204, -0.12240042, -0.052881263, 0.14504352, -0.027575849, -0.029159836, 0.056948386, -0.22972484, 0.048083518, -0.08283192, -0.08870137, -0.13305539, -0.14800052, 0.11209895, 0.018417332, 0.06106651, -0.1565837, -0.04531728, -9.4483905e-05, -0.15177698, -0.035404112, -0.018615924, -0.086388625, 0.1479552, -0.0691746, -0.11520442, 0.15091951, 0.025320057, -0.040738493, 0.13207342, -0.11437645, 0.12112834, -0.20108114, 0.036148477, 0.16951686, -0.112973556, -0.12558381, -0.15290868, -0.027652266, 0.15955453, 0.057484303, -0.020607438, -0.042973083, 0.18042003, 0.1511703, -0.048201293, 0.005797132, 0.16841042, -0.028377028, 0.07037262, -0.16077352, -0.08344605, 0.19266362, -0.09114433, -0.15325478, -0.098621145, -0.022512035, 0.027415577, -0.082386084, 0.06950767, -0.025508331, -0.0725574, -0.21574844, 0.062588006, -0.1538586, 0.010661454, 0.15113872, 0.090077706, 0.11646439, 0.10458184, -0.19077316, -0.22324957, 0.024101902, 0.04935713, -0.03043297, -0.11523694, 0.152684, 0.17141977, 0.14689718, 0.15325002, -0.07688493, -0.06164824, -0.05162103, 0.01665408, 0.17450576, 0.2219197, -0.15646376, 0.04712534, 0.014899327, -0.1527895, -0.22283709, 0.07602442, -0.015619105, 0.20781185, -0.04507689, -0.22379853, 0.067766346, -0.23211081, -0.26482263, -0.1368439, -0.10525755, -0.20353372, -0.20817631, 0.073841475, 0.02099486, -0.15009353, 0.10424877, 0.0057322755, -0.19460154, 0.17533676, -0.035950743, -0.07969183, 0.19085202, 0.15221699, -0.19640653, 0.10774435, -0.11765367, 0.22298118, 0.07812405, 0.097585395, -0.15429275, 0.16278104, -0.035144936, -0.06821058, 0.06878393, -0.21769007, 0.0392155, -0.11170942, 0.10584955, 0.105673164, 0.17082134, -0.17778033, -0.20750292, 0.06309613, 0.12975712, -0.03966571, 0.0062393057, 0.081665464, -0.13586544, -0.097098656, -0.06532861, -0.08166256, -0.08078185, -0.021037, 0.20753613, -0.1287936, -0.17274934, -0.047397025, 0.0013637027};
	static INPUT_DTYPE dense_bias[DENSE_OUTPUT_NODES] = {-0.032419156, 0.03462036, -0.029802812};

	static CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];
	static CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH];
	static DENSE_OUTPUT_DTYPE dense_output_buffer[DENSE_OUTPUT_NODES];

	static int CNN_output_free = CNN_OUTPUT_LENGTH;

	MAIN_PROCESS: if (function_select == 0) {
		// input more data
		memcpy(input_buffer, data_in, sizeof(CNN_RAW_IN_DTYPE) * CNN_KERNEL_LENGTH*INPUT_DEPTH);
		compute_convolution(input_buffer, CNN_weights, CNN_bias, cnn_output_buffer);
		memcpy(cnn_output, cnn_output_buffer, sizeof(CNN_OUT_DTYPE) * CNN_OUTPUT_LENGTH*CNN_OUTPUT_DEPTH);
		// compute dense, need software check if result is valid (data_required == 0)
		compute_dense(cnn_output_buffer, dense_weights, dense_bias, dense_output_buffer);
		memcpy(raw_output, dense_output_buffer, sizeof(DENSE_OUTPUT_DTYPE) * DENSE_OUTPUT_NODES);
		argmax(dense_output_buffer, result_out);
		CNN_output_free = (CNN_output_free == 0) ? 0 : CNN_output_free - 1;
		data_required = CNN_output_free;
	} else if (function_select == 1) {
		// read raw results from CNN
		// memcpy(cnn_output, cnn_output_buffer, sizeof(CNN_OUT_DTYPE) * CNN_OUTPUT_LENGTH*CNN_OUTPUT_DEPTH);
	} else if (function_select == 2) {
		// reset CNN output buffer
		memset(cnn_output_buffer, 0, sizeof(CNN_OUT_DTYPE) * CNN_OUTPUT_LENGTH*CNN_OUTPUT_DEPTH);

		// reset input buffer
		memset(input_buffer, 0, sizeof(CNN_IN_DTYPE) * CNN_KERNEL_LENGTH*INPUT_DEPTH);

		// reset the number of data required
		CNN_output_free = CNN_OUTPUT_LENGTH;
		data_required = CNN_output_free;
	} else if (function_select == 3) {
		// set CNN weights
		memcpy(CNN_weights, weights_and_bias, sizeof(CNN_WEIGHTS_DTYPE) * CNN_KERNEL_LENGTH*CNN_KERNEL_DEPTH*CNN_KERNEL_COUNT);
	} else if (function_select == 4) {
		// set CNN bias
		memcpy(CNN_bias, weights_and_bias, sizeof(CNN_BIAS_DTYPE) * CNN_KERNEL_COUNT);
	} else if (function_select == 5) {
		// set dense weights
		memcpy(dense_weights, weights_and_bias, sizeof(DENSE_WEIGHTS_DTYPE) * DENSE_INPUT_NODES*DENSE_OUTPUT_NODES);
	} else if (function_select == 6) {
		// set dense bias
		memcpy(dense_bias, weights_and_bias, sizeof(DENSE_BIAS_DTYPE) * DENSE_OUTPUT_NODES);
	} else {
		// do nothing
	}
}
