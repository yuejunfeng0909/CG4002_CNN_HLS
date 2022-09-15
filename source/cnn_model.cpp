#include "cnn_model.h"
#include <string.h>
#include "set_weight_bias.h"

// set up stream
// #include "ap_int.h"
//#include "hls_stream.h"
//
// typedef ap_int<16> int16_t;  // 16-bit raw data width
//hls::stream<uint16_t> data_in_stream;  // A stream declaration

void cnn_action_detection(
		FUNCTION_SELECT_BIT_WIDTH function_select,
		CNN_RAW_IN_DTYPE *data_in,
		int &result_out) {
#pragma HLS INTERFACE mode=s_axilite port=function_select
#pragma HLS INTERFACE mode=m_axi port=data_in depth=75*6
#pragma HLS INTERFACE mode=s_axilite port=result_out
#pragma HLS interface s_axilite port=return bundle=control

static INPUT_DTYPE CNN_weights[CNN_KERNEL_LENGTH][CNN_KERNEL_DEPTH][CNN_KERNEL_COUNT] = {-0.048345223, 0.08655253, 0.059436936, -0.15124185, 0.104241766, -0.044725608, -0.05378127, 0.15561302, 0.13968457, 0.030940533, -0.09337533, -0.10085036, -0.12493748, 0.023272626, 0.039146055, 0.0003685956, 0.0070452048, -0.09566883, -0.030648109, 0.0014461267, 0.0374296, -0.08092408, -0.11544786, 0.15825264, -0.024858586, -0.033007797, 0.10198778, 0.06583119, -0.016747909, 0.08299139, 0.057019018, 0.06944271, 0.012257936, -0.12860765, 0.041344665, 0.15855078, 0.101339065, 0.07409855, 0.1599809, -0.13408826, 0.13876632, 0.039479658, -0.04030006, -0.14656712, 0.13888271, -0.0492974, 0.077358514, 0.1212742, 0.013365944, 0.10828862, -0.025789615, -0.080948845, 0.083394356, -0.025713665, 0.112958655, -0.0973463, 0.056653094, -0.09171473, -0.02735439, 0.034496635, 0.11127055, -0.04425766, -0.07457639, -0.09288928, 0.096196406, 0.09591489, 0.010231059, 0.05772008, -0.06786689, -0.06088417, 0.11870133, 0.055623528, -0.119509354, 0.002460557, 0.07801786, 0.14298777, 0.15137757, 0.022650234, -0.08085264, -0.029504249, -0.08982712, -0.012581627, 0.039720252, -0.0684025, -0.0098773865, 0.08327919, 0.14198606, 0.15115955, -0.11253653, 0.13383825, 0.061027255, 0.15000886, 0.15217514, -0.09898937, -0.08133686, 0.09682455, 0.0662932, 0.0032717467, -0.074763834, 0.10718007, 0.13891082, 0.008361681, 0.009617994, -0.049513746, 0.01865597, -0.1506908, 0.03552902, 0.03464264, -0.05931303, 0.12524281, -0.09543877, -0.107990876, -0.11081034, 0.1427921, 0.060833644, 0.014122025, -0.1397462, 0.09035932, -0.057090793, -0.0633742, 0.071631595, -0.108393155, -0.15586832, -0.018934662, -0.11930715, -0.03644776, 0.036338545, -0.002807379, -0.096612744, -0.122227415, 0.13499153, -0.0898109, -0.03423636, 0.16320851, -0.0033828716, -0.11650933, -0.10771859, 0.1297737, -0.08877244, -0.04719223, 0.08477132, -0.021031411, -0.1721936, 0.12676278, 0.11104063, 0.09794234, 0.15292288, 0.14629737, 0.14268973, -0.10526123, 0.0031011873, 0.095977746, 0.104543336, -0.1465132, -0.15741213, 0.054047454, -0.062554225, -0.11891017, -0.046011075, -0.14028712, 0.046155974, -0.122332655, 0.072531074, 0.14154652, -0.045865476, -0.022901729, -0.066736855, -0.031259928, 0.07037137, 0.087381415, -0.020191958, 0.012334693, -0.09463218, 0.003989589, 0.13307515, -0.013690259, 0.055350795, 0.06053561, 0.13261351, -0.028973209, 0.12686937, 0.105069436, -0.071782514, 0.12688367, -0.077126205, -0.0728868, -0.13344015, 0.095439, 0.09550721, 0.019485913, -0.10492496, 0.029328994, 0.07680806, 0.047437806, -0.0625678, 0.12359059, -0.055304185, 0.0775465, -0.06459921, -0.014717335, 0.005702533, -0.1274106, -0.046690654, 0.05883767, -0.10098523, -0.08065987, -0.035423342, 0.01907457, 0.043981712, -0.0951025, 0.012721885, -0.14230002, -0.14698221, -0.032934535, 0.085995406, -0.14215603, 0.14128588, 0.007710763, -0.025329666, -0.03560199, -0.1388605, 0.14270037, 0.023813367, -0.080975525, 0.069288515, 0.0965692, 0.14020705, 0.11478266, 0.033252787, -0.11448044, 0.0182266, -0.052009895, -0.09614832, 0.04899736, 0.052704453, -0.14867641, 0.010189907, 0.060615856, -0.059464447, -0.057611845, -0.040288594, 0.05852618, -0.10819052, -0.022016214, 0.10218447, -0.018954428, -0.010901632, 0.015468117, -0.14837669, 0.092467286, -0.08635428, -0.04736027, -0.07245877, -0.12547578, 0.0966098, -0.0249992, 0.10378423, -0.10813952, -0.035883628, 0.005055959, -0.08958344, 0.024496201, 0.008796042, -0.10731252, 0.11163501, -0.045666642, -0.044688396, 0.15748309, -0.005874236, 0.0003512158, 0.105827585, -0.08867627, 0.08136855, -0.15136564, -0.0030547127, 0.15219547, -0.06290058, -0.13788685, -0.051204372, 0.0436083, 0.09114963, 0.049797375, 0.035799775, -0.018416509, 0.10774831, 0.07105125, 0.00844401, -0.0047446736, 0.11915562, -0.0066775, 0.05035734, -0.0884357, -0.025508288, 0.100710064, -0.04379789, -0.11326875, -0.072676376, 0.034134146, -0.03151289, -0.028894199, 0.047234178, -0.014078501, -0.062435415, -0.13537155, 0.034110356, -0.11123936, 0.05367478, 0.06625671, 0.10443436, 0.13560405, 0.06156445, -0.114566974, -0.14852497, 0.033847358, 0.024991283, -0.06697051, -0.029134506, 0.14374156, 0.14708492, 0.09944718, 0.13917953, -0.026098777, -0.026485743, -0.07046995, 0.039986935, 0.13932067, 0.16512924, -0.06538425, 0.010473833, 0.042747412, -0.13694721, -0.14993258, 0.049977213, 0.02068017, 0.11700973, -0.012710004, -0.15093763, 0.02449131, -0.12240585, -0.14049986, -0.13272732, -0.07521487, -0.115798764, -0.12710436, 0.035744544, -0.03588188, -0.06576373, 0.072314, 0.011539579, -0.13994695, 0.13274197, -0.049036756, -0.030055085, 0.12002872, 0.15180372, -0.15750343, 0.088523336, -0.040663198, 0.11136409, 0.09620915, 0.09517286, -0.11070233, 0.09460039, 0.01751558, -0.08574245, 0.07623709, -0.11870993, -0.009042423, -0.053949405, 0.13433929, 0.07343962, 0.123357944, -0.13649215, -0.1584166, 0.0653964, 0.052746065, 0.040699188, 0.012031635, 0.07591685, -0.095437825, -0.063565545, -0.04875351, -0.033770252, -0.09649554, 0.013708706, 0.15990606, -0.08664291, -0.089754134, -0.055747245, 0.03302777, 0.15220174, -0.026321873, 0.10709876, 0.005857435, 0.13644505, 0.053121455, -0.101122364, 0.13820888, -0.018597625, -0.09380622, 0.060787767, 0.06419058, 0.13452248, 0.12173209, -0.08447487, -0.011839177, 0.08272609, 0.08250371, 0.07742924, 0.08463337, 0.12003576, 0.103800036, 0.035995603, -0.15748532, 0.057305127, 0.014170753, 0.053722188, -0.02261447, -0.03139472, -0.079784885, 0.09159798, -0.0042839125, 0.020743812, 0.034936406, 0.11184991, 0.13383871, 0.13269696, 0.12575139, -0.09745569, 0.060227375, -0.08442664, -0.041094936, -0.15079814, 0.06160297, 0.08043344, 0.14200181, -0.08468884, 0.053086463, -0.11649875, -0.09784237, 0.17294051, 0.0443014, -0.118874446, 0.15298568, 0.06342243, 0.11985265, -0.07986345, -0.040170852, -0.08966637, 0.08794173, -0.08828819, 0.14069597, 0.037401397, 0.032182224, 0.102148965, -0.019534469, -0.09031028, 0.006744431, 0.167738, -0.048605707, 0.15385388, 0.0006716341, -0.057879742, 0.122423045, -0.012230617, 0.020585138, 0.016177634, 0.06561342, 0.036800265, 0.07118369, -0.150473, 0.081571706, 0.031774677, 0.12575734, -0.008856822, 0.046544097, -0.112921536, 0.11136357, -0.051916536, 0.06273395, -0.12231484, 0.09674687, 0.14058572, -0.13590077, -0.13368122, 0.08444575, 0.12485947, -0.07573446, 0.022688128, -0.10634961, -0.1067017, 0.009958783, 0.03707399, -0.0645463, -0.08293576, 0.024736878, -0.03050624, -0.04066419, 0.15597436, -0.0034479192, 0.15459378, 0.0985366, 0.06610827, 0.03598178, 0.03175633, 0.058976814, -0.11301711, 0.04157054, -0.06833801, 0.09078117, -0.10930277, 0.07810803, 0.12597986, -0.13983667, -0.08404276, -0.08853826, 0.009513726, 0.028141545, 0.15962227, -0.12969969, 0.06308797, 0.12540628, -0.04747224, -0.08391128, 0.079305634, 0.071690105, 0.15054205, -0.068411954, -0.12807347, -0.11232665, -0.0786561, 0.022510162, 0.023465557, -0.120564006, 0.06604043, -0.13895506, 0.01884844, 0.16322039, -0.16168173, -0.14368188, 0.10600945, 0.124610886, 0.11814405, 0.12222186, 0.14562343, 0.10995694, -0.14357097, -0.11768407, 0.0815508, -0.07613714, 0.0902306, -0.0668424, 0.11797776, -0.031427074, 0.10134561, -0.14144024, 0.099600784, -0.09962457, -0.10352151, 0.13098893, 0.020601375, -0.12108719, 0.11512147, 0.027497325, -0.029981432, -0.02741966, 0.07248068, 0.108985275, -0.12373534, -0.07267364, -0.035625897, 0.008443906, -0.122584246, -0.03199765, 0.013595391, -0.115355, -0.1029985, -0.04980309, -0.13857399, 0.09920672, -0.097281866, 0.108760744, -0.09017842, -0.08869979, 0.13429026, 0.053010162, 0.14751486, 0.08231978, 0.11382711, 0.048877887, -0.025702609, 0.046566375, 0.06825422, 0.04588729, 0.05827328, -0.121347554, 0.007259601, 0.016044483, 0.020948129, 0.09116253, -0.080223806, -0.0032689208, -0.13102518, -0.025646243, -0.09308545, -0.026283005, 0.05590045, 0.13487509, -0.0042329216, -0.1294723, -0.1387116, 0.071121864, -0.071859606, -0.07250233, -0.0661744, -0.15031381, 0.009584285, 0.065578766, 0.13370778, -0.09875599, -0.13377279, 0.050552707, -0.119455494, 0.13548644, -0.041078072, 0.053788714, -0.104271166, 0.13227743, -0.07849556, -0.1325929, -0.086112656, 0.01817082, -0.026543718, -0.028661603, 0.1463638, -0.11680549, -0.053779, -0.0002869991, 0.11415998, -0.087676995, -0.1444882, 0.061192747, 0.05623265, -6.1532926e-05, -0.13813126, 0.0715559, -0.13784926, 0.09091471, 0.14029928, -0.09880383, 0.094018154, 0.12739044, -0.107905656, -0.033122018, 0.06569459, -0.0060039135, -0.088585295, -0.09851815, -0.16072777, -0.09546447, -0.028063443, 0.117020994, 0.12639304, -0.026141457, 0.15225345, 0.12836157, 0.1635006, 0.05492548, 0.07280236, -0.021013308, 0.09499061, -0.14280057, -0.0822615, -0.13265327, -0.064961106, 0.1323231, -0.10421753, -0.08526956, -0.10385709, -0.06333755, -0.06909275, -0.14385283, -0.07743845, 0.08848049, -0.07246627, -0.11723555, 0.0931053, 0.12809709, 0.08223194, -0.08097419, -0.067654334, -0.05010565, -0.060149577, -0.15293857, -0.13321942, -0.0077354885, 0.10607569, -0.0147731155, 0.0074291113, -0.080044195, -0.14462617, -0.009168241, -0.091318786, -0.122967884, 0.0013257746, -0.09389704, -0.06279239, 0.059014197, -0.14464717, 0.034490913, -0.14192228, 0.12287527, 0.06493227, -0.04041458, 0.02273571, 0.1205845, 0.030023403, 0.08586455, 0.03658338, -0.014613784, 0.018384112, 0.120917596, 0.04168683, -0.10767826, 0.09361493, -0.03802927, -0.11633401, -0.018670997, 0.021585848, 0.006141859, 0.006971059, -0.007784013, 0.12318079, -0.013717303, 0.09147403, -0.0645341, -0.032793753, 0.05580548, -0.026399951, -0.01587282, 0.13617139, 0.004576308, -0.10154208, 0.08645489, -0.037215073, -0.011940719, 0.0694736, 0.113394365, 0.099950984, -0.118533745, -0.028885137, 0.04616627, -0.07025025, 0.10765315, -0.041206613, -0.09834762, -0.016357157, -0.051088396, 0.044812366, 0.035576705, -0.09649095, 0.00679845, 0.13532552, 0.14069434, 0.14520632, -0.13814247, 0.006491755, -0.08656279, -0.1012912, 0.054036856, 0.0788728, 0.009015649, 0.029360792, 0.044963066, -0.105373874, 0.1363174, 0.025660986, 0.14974955, 0.11244107, -0.15288377, -0.14451696, -0.059763845, 0.09523281, 0.054215346, -0.13322505, 0.08750356, -0.092046306, -0.124857664, 0.07980272, 0.11193036, 0.016626183, 0.12571768, 0.0259034, 0.015423576, -0.11339755, 0.10421655, -0.053533357, 0.14851476, 0.05502978, 0.11816862, -0.07349132, 0.13854116, 0.0810348, 0.10804411, -0.088249266, -0.050428778, 0.00021191918, -0.09135431, 0.0806424, -0.07182745, 0.12860523, 0.0011638732, -0.10818704, 0.1217796, -0.12770429, -0.03153534, 0.13388498, 0.11430126, -0.064444155, 0.16350117, -0.048892885, 0.020436823, 0.03982125, 0.15850905, 0.08940121, -0.065662995, -0.031319838, 0.04380564, 0.04961235, 0.11742945, 0.14314523, 0.0928198, 0.09632106, 0.107674986, -0.09073546, -0.013699214, 0.048997782, -0.074366055, 0.123513915, 0.008329342, 0.08860455, -0.030208947, -0.11819755, -0.13466865, 0.068767324, 0.13127501, 0.124652505, 0.075263605, -0.123312324, -0.05012123, -0.10966204, 0.15213381, -0.12480431, -0.022568393, 0.006371229, 0.12444786, 0.09102174, -0.14002058, -0.0012753685, -0.06798931, 0.043087997, -0.0043468415, 0.060268864, 0.096242756, 0.075567126, -0.040858813, -0.107160054, 0.11895109, 0.040914617, -0.023041595, 0.12316325, -0.14285827, -0.098516256, -0.059407648, -0.14016482, 0.054953557, 0.039087944, -0.04658419, 0.09492843, 0.11897176, 0.07632932, -0.010822609, -0.06164758, -0.09158656, 0.14956641, -0.13085236, 0.062000778, 0.09560043, -0.13440067, 0.023680205, 0.08797973, 0.00610368, 0.09278359, -0.096970275, -0.10300764, 0.0217834, -0.12278715, -0.088522494, -0.087130524, 0.06233345};
static INPUT_DTYPE CNN_bias[CNN_KERNEL_COUNT] = {-0.008858758, -0.016352072, 0.011444908, -0.0018160417, -0.01636437, 0.0033206155, -0.005643273, -0.013887742, -0.01306393, -0.0024265368};
static INPUT_DTYPE dense_weights[DENSE_INPUT_NODES][DENSE_OUTPUT_NODES] = {-0.087586954, 0.14176308, 0.0779845, -0.22016522, 0.13827163, -0.06256831, -0.10168408, 0.22571883, 0.1783986, 0.011188561, -0.13124329, -0.1639552, -0.14477937, 0.022384947, 0.017966818, -0.0010121124, 0.0010113443, -0.15026982, -0.048951454, 0.0061287577, 0.01785696, -0.1278693, -0.12383373, 0.20003003, -0.051053062, -0.038637128, 0.10178256, 0.067760065, -0.026571883, 0.105097026, 0.064949766, 0.11381789, 0.009457986, -0.16847727, 0.079196535, 0.191386, 0.121211894, 0.10263721, 0.18138422, -0.21058989, 0.19698875, 0.05502262, -0.06214182, -0.19974484, 0.16351624, -0.065457955, 0.124384366, 0.1876754, 0.04524736, 0.123721324, -0.044625167, -0.12952268, 0.109365456, -0.04125075, 0.14746608, -0.11522653, 0.047103453, -0.14900443, -0.0059080794, 0.02907376, 0.12332479, -0.030554457, -0.098535255, -0.14211826, 0.12727186, 0.12669085, -0.017788105, 0.09453133, -0.092889674, -0.10845469, 0.15461345, 0.046749853, -0.13935833, 0.0021653622, 0.067811504, 0.18108086, 0.19385336, 0.025350433, -0.110888585, -0.04571127, -0.15411727, -0.037064306, 0.08207373, -0.09865979, -0.01941774, 0.11569143, 0.15343985, 0.17935887, -0.15498467, 0.17294602, 0.07330489, 0.22200839, 0.20576306, -0.13374199, -0.08751077, 0.11221649, 0.085757405, 0.0034935246, -0.13105021, 0.10758869, 0.19466251, 0.031097837, -0.027804306, -0.06523439, 0.0021946013, -0.19490495, 0.059341285, 0.067133665, -0.05558165, 0.13712406, -0.13091104, -0.17733067, -0.14579695, 0.18686815, 0.07577094, 0.032310788, -0.21731666, 0.09212793, -0.04615641, -0.09910768, 0.07209976, -0.11195484, -0.21209754, -0.042338017, -0.16254023, -0.05074663, 0.033267085, 0.011204193, -0.1373326, -0.19354503, 0.17121814, -0.13726123, -0.044099126, 0.21757887, -0.038117733, -0.14823988, -0.16027291, 0.15105295, -0.122948684, -0.055235267, 0.07903506, -0.051040012, -0.20084085, 0.16368629, 0.1364771, 0.13569714, 0.16818725, 0.17544593, 0.18307687, -0.14369906, -0.006935119, 0.14498349, 0.13117182, -0.1924218, -0.1837004, 0.055570256, -0.074061066, -0.15592805, -0.09922799, -0.2222509, 0.06391876, -0.14605837, 0.061436545, 0.18786527, -0.07644547, -0.0096055595, -0.092463344, -0.020564081, 0.11459609, 0.108009316, -0.03175969, -0.016583394, -0.120253064, 0.0061544525, 0.1545679, -0.0013175737, 0.04381782, 0.050132744, 0.20956427, -0.04626022, 0.13969755, 0.17834045, -0.10229509, 0.1496027, -0.10337662, -0.09556878, -0.18154165, 0.1385135, 0.11852324, -0.006748726, -0.14162952, 0.006909826, 0.10341832, 0.065376885, -0.11206091, 0.16887802, -0.10254601, 0.09480484, -0.102855, -0.0094782645, -0.028641405, -0.197651, -0.031927865, 0.07716594, -0.161586, -0.10320629, -0.0848624, 0.002701909, 0.052674808, -0.13231616, 0.0019603788, -0.17850462, -0.20093144, -0.04437335, 0.12363017, -0.18844406, 0.19600068, 0.008577878, -0.06301629, -0.08226071, -0.1866815, 0.20159781, -0.0031323931, -0.11197828, 0.08448637, 0.1591325, 0.18002044, 0.17299403, 0.058583047, -0.16402999, 0.015118265, -0.10095693, -0.12924586, 0.07719427, 0.04503571, -0.18306942, -0.016200764, 0.04718329, -0.043390997, -0.08885802, -0.07862753, 0.111979455, -0.14681204, -0.03473788, 0.122304685, -0.01876282, -0.019771628, 0.029824434, -0.202332, 0.09541628, -0.12048551, -0.09417802, -0.09525686, -0.16028021, 0.101229824, -0.015315207, 0.109359026, -0.1704328, -0.058219388, 0.014660744, -0.1533155, 0.0052714935, 0.020423086, -0.12552662, 0.11946368, -0.0512602, -0.09785321, 0.18533987, -0.013356767, -0.005508247, 0.1422603, -0.10443029, 0.096268296, -0.18576905, 0.012658891, 0.19060533, -0.06554945, -0.18298814, -0.107144274, 0.025247822, 0.12048522, 0.07837385, 0.009351216, -0.028541248, 0.15524706, 0.12775075, 0.019736309, -0.0053725676, 0.16895804, -0.020336786, 0.065812595, -0.1271242, -0.043714166, 0.14483598, -0.09171567, -0.13294435, -0.12530378, 0.010806738, -0.0051074806, -0.05396983, 0.052463382, 0.012321673, -0.093695395, -0.17657849, 0.03856916, -0.14954078, 0.052365966, 0.11342045, 0.12688226, 0.1571442, 0.0750084, -0.16627301, -0.19753632, 0.049491823, 0.013362596, -0.0720847, -0.058796756, 0.16113739, 0.18148755, 0.14332667, 0.15663606, -0.04274237, -0.025017252, -0.09557479, 0.018621005, 0.19588679, 0.19379109, -0.11362598, 0.010463201, 0.04733324, -0.17027402, -0.19339898, 0.061523948, 0.029877087, 0.17995372, -0.028857347, -0.20115477, 0.03615628, -0.20097643, -0.22191666, -0.17330728, -0.09561865, -0.19029886, -0.18274963, 0.04513015, -0.01951745, -0.08114419, 0.10273756, 0.015578148, -0.20224966, 0.18255551, -0.0674509, -0.05175273, 0.1666775, 0.16900061, -0.19188684, 0.09258724, -0.08691255, 0.18607895, 0.11554052, 0.11154582, -0.12298951, 0.12908117, 0.019754978, -0.1100838, 0.09100961, -0.18924579, 0.011544295, -0.084682636, 0.1550636, 0.07644027, 0.15969975, -0.17412245, -0.20096052, 0.057911407, 0.08765384, 0.03180103, -0.0143268015, 0.07720155, -0.123623304, -0.111317456, -0.07135348, -0.04004467, -0.11714834, -0.016762707, 0.22321399, -0.14654377, -0.14832239, -0.07967817, 0.03501997};
static INPUT_DTYPE dense_bias[DENSE_OUTPUT_NODES] = {-0.011413713, 0.016356073, -0.015211569};

static CNN_IN_DTYPE input_buffer[CNN_KERNEL_LENGTH][INPUT_DEPTH];
static CNN_OUT_DTYPE cnn_output_buffer[CNN_OUTPUT_LENGTH][CNN_OUTPUT_DEPTH];
static DENSE_OUTPUT_DTYPE dense_output[DENSE_OUTPUT_NODES];

	static int CNN_output_free = CNN_OUTPUT_LENGTH;

	if (function_select == 0) {
		// input more data
		copy_inputs(data_in, &input_buffer[0][0]);
		compute_convolution(input_buffer, CNN_weights, CNN_bias, cnn_output_buffer);
		CNN_output_free = (CNN_output_free == 0) ? 0 : CNN_output_free - 1;
		// when the CNN output count = dense input count, compute dense
		if (CNN_output_free == 0) {
			compute_dense(cnn_output_buffer, dense_weights, dense_bias, dense_output);
		}

	} else if (function_select == 1) {
		// read result
		argmax(dense_output, result_out);
	} else if (function_select == 2) {
		// reset CNN output buffer
		reset(cnn_output_buffer);
		// reset the CNN output count
		CNN_output_free = CNN_OUTPUT_LENGTH;
	}
}
