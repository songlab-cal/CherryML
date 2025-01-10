#include "../branch_length_estimation.h"
#include "../io_helpers.h"
#include "../types.h"
#include <vector>
#include <string>
#include <iostream>

inline int get_char_index(char c) {
    auto it = default_alphabet.find(c);
    return (it != default_alphabet.end()) ? it->second : -1;
}
std::vector<int> string_to_ints(const std::string& x) {
    std::vector<int> res;
    res.reserve(x.length());
    for(int i = 0; i < x.length(); i++) {
        res.push_back(get_char_index(x[i]));
    }
    return res;
}

void test_branch_lengths(
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries,
    std::vector<int> expected
) {
    std::vector<double> rate_categories = {0.25, 0.7, 1.7, 4.0};
    std::vector<int> site_to_rate = {0, 3, 0, 3, 3, 3, 0, 3};
    std::vector<double> quantization_points = {6.729602379904665e-05, 7.402562617895133e-05, 8.142818879684645e-05, 8.95710076765311e-05, 9.852810844418422e-05, 0.00010838091928860265, 0.00011921901121746292, 0.00013114091233920925, 0.0001442550035731302, 0.0001586805039304432, 0.00017454855432348754, 0.0001920034097558363, 0.00021120375073141996, 0.00023232412580456198, 0.00025555653838501814, 0.00028111219222352006, 0.000309223411445872, 0.0003401457525904593, 0.00037416032784950526, 0.0004115763606344558, 0.00045273399669790143, 0.0004980073963676917, 0.0005478081360044608, 0.0006025889496049069, 0.0006628478445653976, 0.0007291326290219376, 0.0008020458919241313, 0.0008822504811165445, 0.0009704755292281991, 0.0010675230821510192, 0.001174275390366121, 0.0012917029294027332, 0.0014208732223430067, 0.0015629605445773075, 0.0017192565990350385, 0.0018911822589385424, 0.0020803004848323967, 0.0022883305333156363, 0.0025171635866472, 0.0027688799453119205, 0.003045767939843113, 0.0033503447338274245, 0.0036853792072101673, 0.004053917127931184, 0.004459308840724303, 0.004905239724796734, 0.005395763697276408, 0.005935340067004049, 0.0065288740737044545, 0.0071817614810749004, 0.007899937629182391, 0.00868993139210063, 0.009558924531310695, 0.010514816984441766, 0.011566298682885941, 0.012722928551174538, 0.013995221406291992, 0.015394743546921193, 0.016934217901613313, 0.018627639691774646, 0.020490403660952113, 0.022539444027047325, 0.024793388429752063, 0.02727272727272727, 0.03, 0.033, 0.036300000000000006, 0.039930000000000014, 0.04392300000000001, 0.04831530000000001, 0.05314683000000002, 0.058461513000000034, 0.06430766430000004, 0.07073843073000005, 0.07781227380300007, 0.08559350118330007, 0.09415285130163009, 0.1035681364317931, 0.11392495007497243, 0.12531744508246967, 0.13784918959071665, 0.15163410854978834, 0.1667975194047672, 0.18347727134524391, 0.2018249984797683, 0.22200749832774516, 0.24420824816051973, 0.2686290729765717, 0.2954919802742289, 0.3250411783016518, 0.357545296131817, 0.3932998257449988, 0.43262980831949865, 0.47589278915144856, 0.5234820680665935, 0.5758302748732529, 0.6334133023605781, 0.6967546325966361, 0.7664300958562997, 0.8430731054419298, 0.9273804159861229, 1.020118457584735, 1.1221303033432088, 1.2343433336775298, 1.357777667045283, 1.4935554337498114, 1.6429109771247925, 1.807202074837272, 1.9879222823209994, 2.1867145105530996, 2.4053859616084097, 2.6459245577692507, 2.910517013546176, 3.201568714900794, 3.5217255863908736, 3.8738981450299614, 4.261287959532957, 4.687416755486254, 5.15615843103488, 5.671774274138368, 6.238951701552206, 6.8628468717074265, 7.54913155887817, 8.304044714765988, 9.134449186242586, 10.047894104866845, 11.052683515353532, 12.157951866888887, 13.373747053577777,20,30,40,50,60};
    transition_matrices log_transition_matrices = read_rate_compute_log_transition_matrices("tests/lg.txt", quantization_points, rate_categories, 20);
    
    std::vector<std::vector<int>> valid_indices_for_branch_length(cherries.size());
    for (int cherry_index = 0; cherry_index < cherries.size(); cherry_index++) {
        const std::vector<int>& x = cherries[cherry_index].first;
        const std::vector<int>& y = cherries[cherry_index].second;
        // Create a list of valid pairs
        std::vector<int> valid_indices;
        valid_indices.reserve(x.size());
        for (int i = 0; i < x.size(); i++) {
            if (x[i] != -1 && y[i] != -1) {
                valid_indices_for_branch_length[cherry_index].push_back(i);  // Store index of valid pair
            }
        }
    }
    std::vector<int> lengths = get_branch_lengths(
        cherries, 
        log_transition_matrices,
        quantization_points,
        site_to_rate,
        valid_indices_for_branch_length
    );
    for(int i = 0; i < cherries.size(); i++) {
        assert(lengths[i] == expected[i]);
    }
    std::cout << "test branch lengths passes" << std::endl;
}

void test_branch_lengths1() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,1,0,1,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,0,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,1,0,1,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,1,2,1}, {0,0,0,0,0,0,0,0}}
    };
    std::vector<int> expected = {103, 81, 90, 103, 109};
    test_branch_lengths(cherries, expected);
}

void test_branch_lengths2() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,1,0,1,1,1,0,1}, {0,1,0,0,0,0,0,0}},
        {{0,0,0,0,0,1,0,1}, {0,1,0,0,3,0,2,0}},
        {{0,0,0,0,1,0,0,1}, {0,0,0,0,3,0,2,0}},
        {{0,1,0,1,1,1,0,1}, {0,0,0,0,0,1,5,0}},
        {{0,0,0,0,1,1,2,1}, {0,0,0,0,0,0,5,0}}
    };
    std::vector<int> expected = {98, 110, 89, 103, 103};
    test_branch_lengths(cherries, expected);
}
void test_branch_lengths3() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,0,0,0,0,0,0,0}, {0,1,0,0,0,0,0,0}},
        {{0,0,0,0,0,1,0,1}, {0,1,0,0,3,0,2,0}},
        {{0,0,0,0,0,0,5,0}, {0,0,0,0,3,0,2,0}},
        {{2,3,0,1,0,1,0,0}, {0,0,0,0,0,1,5,0}},
        {{0,5,0,0,1,1,2,0}, {0,0,0,0,0,0,5,0}}
    };
    std::vector<int> expected = {71, 110, 79, 120, 94};
    test_branch_lengths(cherries, expected);
}
void test_get_site_rates(
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries,
    std::vector<int> expected
) {
    std::vector<double> rate_categories = {0.25, 0.6, 1.7, 4.0};
    std::vector<double> quantization_points = {6.729602379904665e-05, 7.402562617895133e-05, 8.142818879684645e-05, 8.95710076765311e-05, 9.852810844418422e-05, 0.00010838091928860265, 0.00011921901121746292, 0.00013114091233920925, 0.0001442550035731302, 0.0001586805039304432, 0.00017454855432348754, 0.0001920034097558363, 0.00021120375073141996, 0.00023232412580456198, 0.00025555653838501814, 0.00028111219222352006, 0.000309223411445872, 0.0003401457525904593, 0.00037416032784950526, 0.0004115763606344558, 0.00045273399669790143, 0.0004980073963676917, 0.0005478081360044608, 0.0006025889496049069, 0.0006628478445653976, 0.0007291326290219376, 0.0008020458919241313, 0.0008822504811165445, 0.0009704755292281991, 0.0010675230821510192, 0.001174275390366121, 0.0012917029294027332, 0.0014208732223430067, 0.0015629605445773075, 0.0017192565990350385, 0.0018911822589385424, 0.0020803004848323967, 0.0022883305333156363, 0.0025171635866472, 0.0027688799453119205, 0.003045767939843113, 0.0033503447338274245, 0.0036853792072101673, 0.004053917127931184, 0.004459308840724303, 0.004905239724796734, 0.005395763697276408, 0.005935340067004049, 0.0065288740737044545, 0.0071817614810749004, 0.007899937629182391, 0.00868993139210063, 0.009558924531310695, 0.010514816984441766, 0.011566298682885941, 0.012722928551174538, 0.013995221406291992, 0.015394743546921193, 0.016934217901613313, 0.018627639691774646, 0.020490403660952113, 0.022539444027047325, 0.024793388429752063, 0.02727272727272727, 0.03, 0.033, 0.036300000000000006, 0.039930000000000014, 0.04392300000000001, 0.04831530000000001, 0.05314683000000002, 0.058461513000000034, 0.06430766430000004, 0.07073843073000005, 0.07781227380300007, 0.08559350118330007, 0.09415285130163009, 0.1035681364317931, 0.11392495007497243, 0.12531744508246967, 0.13784918959071665, 0.15163410854978834, 0.1667975194047672, 0.18347727134524391, 0.2018249984797683, 0.22200749832774516, 0.24420824816051973, 0.2686290729765717, 0.2954919802742289, 0.3250411783016518, 0.357545296131817, 0.3932998257449988, 0.43262980831949865, 0.47589278915144856, 0.5234820680665935, 0.5758302748732529, 0.6334133023605781, 0.6967546325966361, 0.7664300958562997, 0.8430731054419298, 0.9273804159861229, 1.020118457584735, 1.1221303033432088, 1.2343433336775298, 1.357777667045283, 1.4935554337498114, 1.6429109771247925, 1.807202074837272, 1.9879222823209994, 2.1867145105530996, 2.4053859616084097, 2.6459245577692507, 2.910517013546176, 3.201568714900794, 3.5217255863908736, 3.8738981450299614, 4.261287959532957, 4.687416755486254, 5.15615843103488, 5.671774274138368, 6.238951701552206, 6.8628468717074265, 7.54913155887817, 8.304044714765988, 9.134449186242586, 10.047894104866845, 11.052683515353532, 12.157951866888887, 13.373747053577777};
    transition_matrices log_transition_matrices = read_rate_compute_log_transition_matrices("tests/lg.txt", quantization_points, rate_categories, 20);

    std::vector<double> priors;
    priors.reserve(rate_categories.size());
    for(double rate:rate_categories) {
        priors.push_back(2*std::log(rate) - 3*rate);
    }
    std::vector<int> lengths_index = {60, 20, 40, 60, 20, 40, 60, 20, 40, 60, 20, 40};
    
    std::vector<std::vector<int>> valid_indices_for_site_rate(cherries[0].first.size());
    for (int site_index = 0; site_index < cherries[0].first.size(); site_index++) {
        for (int i = 0; i < cherries.size(); i++) {
            int xi = cherries[i].first[site_index];
            int yi = cherries[i].second[site_index];
            if (xi != -1 && yi != -1) {
                valid_indices_for_site_rate[site_index].push_back(i);  // Store valid index for this site
            }
        }
    }

    std::vector<int> site_to_rate = get_site_rates(
        cherries, 
        log_transition_matrices,
        lengths_index,
        priors,
        valid_indices_for_site_rate
    );
    for(int i=0;i<site_to_rate.size();i++) {
        assert(expected[i] == site_to_rate[i]);
    }
    std::cout << "get site rate passes" << std::endl;
}

void test_get_site_rates1() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,1,0,1,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,0,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,1,0,1,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,1,1,1}, {0,0,0,0,0,0,0,0}},
        {{0,1,0,1,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,0,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,1,0,1,1,1,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,1,0,1}, {0,0,0,0,0,0,0,0}}
    };
    std::vector<int> expected = {1, 3, 1, 3, 3, 3, 2, 3};
    test_get_site_rates(
        cherries,
        expected
    );
}
void test_get_site_rates2() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,1,0,0,1,1,0,0}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,0,1,0,0}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,1,0,0}, {0,0,0,0,0,0,0,0}},
        {{0,1,0,0,1,1,0,0}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,0,1,1}, {0,0,0,0,0,0,0,0}},
        {{0,1,0,0,1,0,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,0,0,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,0,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,1,0,0,1,0,0,1}, {0,0,0,0,0,0,0,0}},
        {{0,0,0,0,1,1,0,0}, {0,0,0,0,0,0,0,0}}
    };
    std::vector<int> expected = {1, 3, 1, 1, 3, 3, 2, 3};
    test_get_site_rates(
        cherries,
        expected
    );
}
void test_get_site_rates3() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,1,0,0,1,1,0,2}, {1,0,0,0,0,0,0,2}},
        {{0,0,0,1,0,1,0,2}, {0,0,0,1,0,0,0,2}},
        {{0,0,0,1,1,1,0,2}, {0,0,0,1,0,0,0,2}},
        {{1,1,0,1,1,1,0,2}, {1,0,0,1,0,0,0,2}},
        {{0,0,0,0,1,0,1,1}, {0,0,0,0,0,0,0,1}},
        {{0,1,0,0,1,0,0,1}, {0,0,0,0,0,0,0,1}},
        {{1,0,0,0,0,0,0,1}, {1,0,0,0,0,0,0,1}},
        {{0,0,0,0,1,0,0,1}, {0,0,0,0,0,0,0,1}},
        {{0,1,0,0,1,0,0,1}, {0,0,0,0,0,0,0,1}},
        {{1,0,0,0,1,1,0,0}, {1,0,0,0,0,0,0,0}}
    };
    std::vector<int> expected = {2, 3, 1, 1, 3, 3, 2, 1};
    test_get_site_rates(
        cherries,
        expected
    );
}
void test_get_site_rates4() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,1,0,0,1,1,0,2}, {1,1,0,0,1,0,0,2}},
        {{0,0,0,1,0,1,0,2}, {0,0,0,1,0,0,0,2}},
        {{5,5,5,5,5,5,0,2}, {0,5,0,1,5,0,0,2}},
        {{1,1,0,1,1,1,0,2}, {1,1,0,1,1,0,0,2}},
        {{0,0,0,0,1,0,1,1}, {0,0,0,0,1,0,0,1}},
        {{0,1,0,0,1,0,0,1}, {0,1,0,0,1,0,0,1}},
        {{1,0,0,0,0,0,0,1}, {1,0,0,0,0,0,0,1}},
        {{0,0,0,0,1,0,0,1}, {0,0,0,0,1,0,0,1}},
        {{0,1,0,0,1,0,0,1}, {0,1,0,0,1,0,0,1}},
        {{1,0,0,0,1,1,0,0}, {1,0,0,0,1,0,0,0}}
    };
    std::vector<int> expected = {2, 1, 2, 2, 1, 3, 2, 1};
    test_get_site_rates(
        cherries,
        expected
    );
}

void test_get_site_rates5() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}}
    };
    std::vector<int> expected = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
    test_get_site_rates(
        cherries,
        expected
    );
}
void test_get_site_rates6() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries = {
        {{0,1,2,3,4,5,6,7,0,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,0,4,5,6,7,8,9,10,11,12,0,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,0,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,0,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,0,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,0,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}},
        {{0,1,2,3,4,5,6,7,8,9,0,11,12,13,14,15,16,17,18,19}, {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}}
    };
    std::vector<int> expected = {1, 1, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 1, 1, 1, 1, 2, 1};
    test_get_site_rates(
        cherries,
        expected
    );
}



// copied from FastTree 2.1
double LnGamma (double alpha)
{
/* returns ln(gamma(alpha)) for alpha>0, accurate to 10 decimal places.
   Stirling's formula is used for the central polynomial part of the procedure.
   Pike MC & Hill ID (1966) Algorithm 291: Logarithm of the gamma function.
   Communications of the Association for Computing Machinery, 9:684
*/
   double x=alpha, f=0, z;
   if (x<7) {
      f=1;  z=x-1;
      while (++z<7)  f*=z;
      x=z;   f=-(double)log(f);
   }
   z = 1/(x*x);
   return  f + (x-0.5)*(double)log(x) - x + .918938533204673
	  + (((-.000595238095238*z+.000793650793651)*z-.002777777777778)*z
	       +.083333333333333)/x;
}

// copied from FastTree 2.1
double IncompleteGamma(double x, double alpha, double ln_gamma_alpha)
{
/* returns the incomplete gamma ratio I(x,alpha) where x is the upper
	   limit of the integration and alpha is the shape parameter.
   returns (-1) if in error
   ln_gamma_alpha = ln(Gamma(alpha)), is almost redundant.
   (1) series expansion     if (alpha>x || x<=1)
   (2) continued fraction   otherwise
   RATNEST FORTRAN by
   Bhattacharjee GP (1970) The incomplete gamma integral.  Applied Statistics,
   19: 285-287 (AS32)
*/
   int i;
   double p=alpha, g=ln_gamma_alpha;
   double accurate=1e-8, overflow=1e30;
   double factor, gin=0, rn=0, a=0,b=0,an=0,dif=0, term=0, pn[6];

   if (x==0) return (0);
   if (x<0 || p<=0) return (-1);

   factor=(double)exp(p*(double)log(x)-x-g);
   if (x>1 && x>=p) goto l30;
   /* (1) series expansion */
   gin=1;  term=1;  rn=p;
 l20:
   rn++;
   term*=x/rn;   gin+=term;

   if (term > accurate) goto l20;
   gin*=factor/p;
   goto l50;
 l30:
   /* (2) continued fraction */
   a=1-p;   b=a+x+1;  term=0;
   pn[0]=1;  pn[1]=x;  pn[2]=x+1;  pn[3]=x*b;
   gin=pn[2]/pn[3];
 l32:
   a++;  b+=2;  term++;   an=a*term;
   for (i=0; i<2; i++) pn[i+4]=b*pn[i+2]-an*pn[i];
   if (pn[5] == 0) goto l35;
   rn=pn[4]/pn[5];   dif=fabs(gin-rn);
   if (dif>accurate) goto l34;
   if (dif<=accurate*rn) goto l42;
 l34:
   gin=rn;
 l35:
   for (i=0; i<4; i++) pn[i]=pn[i+2];
   if (fabs(pn[4]) < overflow) goto l32;
   for (i=0; i<4; i++) pn[i]/=overflow;
   goto l32;
 l42:
   gin=1-factor*gin;

 l50:
   return (gin);
}

// copied from FastTree 2.1
double PGamma(double x, double alpha)
{
  /* scale = 1/alpha */
  return IncompleteGamma(x*alpha,alpha,LnGamma(alpha));
}

std::vector<double> get_weights_for_initial_site_rates(
    const std::vector<double>& rate_categories
) {
    // get the cuttoffs for each bin based on the weight for that rate category. 
    // the weight of the curve is the cdf of gamma(shape=3, scale=1/3) from the midpoint of the previous
    // category to the midpoint of the next
    std::vector<double> midpoints;
    midpoints.reserve(rate_categories.size() - 1);
    for(int i = 1; i < rate_categories.size(); i++) {
        midpoints.push_back(sqrt(rate_categories[i-1]*rate_categories[i]));
    }
    std::vector<double> weights;
    double shape = 3.0;
    weights.reserve(rate_categories.size());
    for(int i = 0; i < rate_categories.size() - 1; i++) {
        weights.push_back(PGamma(midpoints[i], shape));
    }
    weights.push_back(1.0);
    
    weights[0] = weights[0];
    for(int r = 1; r < rate_categories.size(); r++) {
        weights[r] = weights[r];
    }
    return weights;
}

void test_lg_ble(
    const std::vector<std::pair<std::vector<int>, std::vector<int> > >& cherries,
    const std::vector<double>& expected_lengths
) {
    std::vector<double> quantization_points = {6.729602379904665e-05, 7.402562617895133e-05, 8.142818879684645e-05, 8.95710076765311e-05, 9.852810844418422e-05, 0.00010838091928860265, 0.00011921901121746292, 0.00013114091233920925, 0.0001442550035731302, 0.0001586805039304432, 0.00017454855432348754, 0.0001920034097558363, 0.00021120375073141996, 0.00023232412580456198, 0.00025555653838501814, 0.00028111219222352006, 0.000309223411445872, 0.0003401457525904593, 0.00037416032784950526, 0.0004115763606344558, 0.00045273399669790143, 0.0004980073963676917, 0.0005478081360044608, 0.0006025889496049069, 0.0006628478445653976, 0.0007291326290219376, 0.0008020458919241313, 0.0008822504811165445, 0.0009704755292281991, 0.0010675230821510192, 0.001174275390366121, 0.0012917029294027332, 0.0014208732223430067, 0.0015629605445773075, 0.0017192565990350385, 0.0018911822589385424, 0.0020803004848323967, 0.0022883305333156363, 0.0025171635866472, 0.0027688799453119205, 0.003045767939843113, 0.0033503447338274245, 0.0036853792072101673, 0.004053917127931184, 0.004459308840724303, 0.004905239724796734, 0.005395763697276408, 0.005935340067004049, 0.0065288740737044545, 0.0071817614810749004, 0.007899937629182391, 0.00868993139210063, 0.009558924531310695, 0.010514816984441766, 0.011566298682885941, 0.012722928551174538, 0.013995221406291992, 0.015394743546921193, 0.016934217901613313, 0.018627639691774646, 0.020490403660952113, 0.022539444027047325, 0.024793388429752063, 0.02727272727272727, 0.03, 0.033, 0.036300000000000006, 0.039930000000000014, 0.04392300000000001, 0.04831530000000001, 0.05314683000000002, 0.058461513000000034, 0.06430766430000004, 0.07073843073000005, 0.07781227380300007, 0.08559350118330007, 0.09415285130163009, 0.1035681364317931, 0.11392495007497243, 0.12531744508246967, 0.13784918959071665, 0.15163410854978834, 0.1667975194047672, 0.18347727134524391, 0.2018249984797683, 0.22200749832774516, 0.24420824816051973, 0.2686290729765717, 0.2954919802742289, 0.3250411783016518, 0.357545296131817, 0.3932998257449988, 0.43262980831949865, 0.47589278915144856, 0.5234820680665935, 0.5758302748732529, 0.6334133023605781, 0.6967546325966361, 0.7664300958562997, 0.8430731054419298, 0.9273804159861229, 1.020118457584735, 1.1221303033432088, 1.2343433336775298, 1.357777667045283, 1.4935554337498114, 1.6429109771247925, 1.807202074837272, 1.9879222823209994, 2.1867145105530996, 2.4053859616084097, 2.6459245577692507, 2.910517013546176, 3.201568714900794, 3.5217255863908736, 3.8738981450299614, 4.261287959532957, 4.687416755486254, 5.15615843103488, 5.671774274138368, 6.238951701552206, 6.8628468717074265, 7.54913155887817, 8.304044714765988, 9.134449186242586, 10.047894104866845, 11.052683515353532, 12.157951866888887, 13.373747053577777};
    std::vector<double> weights = {2.53747333, 2.88345872, 3.17063443, 2.93648371, 4.34766486, 3.19988335, 2.63685675, 2.85881014, 3.80070627, 2.77809284, 2.31181858, 2.73954187, 3.77439477, 3.16292191, 3.12265797, 2.79365811, 2.93206388, 4.4173647, 3.37684729, 2.67152161};
    
    std::vector<std::vector<int> > all_sequences;
    for(std::pair<std::vector<int>, std::vector<int> > c:cherries) {
        all_sequences.push_back(c.first);
        all_sequences.push_back(c.second);
    }

    std::vector<double> rate_categories = {0.25, 0.62996052, 1.58740105, 4.0};
    transition_matrices log_transition_matrices = read_rate_compute_log_transition_matrices("tests/lg.txt", quantization_points, rate_categories, 20);
    
    length_and_rates res = ble(
        cherries,
        all_sequences,
        log_transition_matrices,
        quantization_points,
        rate_categories,
        get_weights_for_initial_site_rates(rate_categories),
        50
    );
    
    for(int i = 0; i < res.lengths.size(); i++) {
        assert(fabs(expected_lengths[i] - res.lengths[i]) < 1e-6 && fabs(expected_lengths[i] - res.lengths[i])/expected_lengths[i] < 1e-6);
    }
    
    std::cout << "lg_ble sanity passes" << std::endl;
}


void test_lg_ble1() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries(4);
    
    std::vector<int> x=string_to_ints("LKTFGHNTVDRVPRIDFYRNTSSLSGVKINRPSLADLHEDLKRDILSVPGTVDGVANG");
    std::vector<int> y=string_to_ints("LQTFGHNTMDAVPKIEYYRNTGSVSGPKVNRPSLQEIHEQLAKNVAVAPGSADRVANG");
    cherries[0] = {x,y};

    x=string_to_ints("LRTFGHNTIDAVPNIDFYRQTAAPLGEKLIRPTLSELHDELDKE-----PFEDGFANG");
    y=string_to_ints("LRTFGHNTMDAVPRIDHYRHTAAQLGEKLLRPSLAELHDELEKE-----PFEDGFANG");
    cherries[1] = {x,y};
    
    x=string_to_ints("VSAFGHDTLDRVPNPDFYRNAASISGHRAVRPSLHELHDVFQKNGGLNLPSPVEDSEG");
    y=string_to_ints("ISAFGHDTLDRVPHIDFYRNAGSMSGHRAVRPSLQELHDVFQKNGAISVPDTLED-DG");
    cherries[2] = {x,y};
    
    x=string_to_ints("MRTFGYNTIDVVPAYEHYANSTQPGEPRKVRPTLADLHSFLKEGRHLHALALDSRPSH");
    y=string_to_ints("MRTFGYNTIDVVPA------STQPGEPRKVRPTLADLHSFLKEGRHLHALALDSRPSH");
    cherries[3] = {x,y};

    std::vector<double> expected_lengths = {1.0201184575847351, 0.3575452961318170, 0.4758927891514486, 0.0000672960237990};

    test_lg_ble(cherries, expected_lengths);
}

void test_lg_ble2() {
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries(5);
    
    std::vector<int> x=string_to_ints("LEWYITKHCLWQYNSRGWDREIQNERILMKTRQILCGESYADRYYWSEAITLAGAFKRYFPWLEDMTKDEIATVMQMLKEKMDYTMIKGSLNLELTKEKY");
    std::vector<int> y=string_to_ints("LVDYIMKNCLWQFNSRGWDRLKQNAGILSQTCEILCGETAMDRCYWVDAVILSRAYKARFPWLMAMTKPEIKSLFKALHEKIDHLTVHGSLNTELTVPHY");
    cherries[0] = {x,y};

    x=string_to_ints("LTDYIMKTLLWQFHSRSWDRERQNAEILKKTKELLCGETSHDRCYWVDAVCLADDYREHYPWINSMSKEEIGSLMQGLKDRMDYLTIHRLLNEELSDKHY");
    y=string_to_ints("LTDYIMKNCLWQFHSRSWDRLRQNEEILKKTKQILCGETNHDRCYWVDAVCMAEDFRADYAWMADLDKEQIASLMDGLYQRINYLTVGGSLNEELTDKNY");
    cherries[1] = {x,y};
    
    x=string_to_ints("PIDFIMKHCLWQSHSRNWDRERQNEEILKKTKQLLCGETPSDRCYWVDAVSLVDAYRERYTWINAMSKDELAQLIDTLKARLDYLTISGSLNEELSDKNY");
    y=string_to_ints("LTDYIMKNCLWQFHSRKWDRERQNEGILTKTKQILLGETPADRCYYADALCLADAYKTEYPWINDMSKDELIQLMQQLKDRIDYVTITGSLNAELTDPRY");
    cherries[2] = {x,y};
    
    x=string_to_ints("LIDYIMKHCLWQFHSRSWDRKRQNEGILTKTTQLLCDETPADKCYWVDAVCLADAYKSRYPWLKTMDKDDIKALMGALHERLDHLTITGSLNLELTDQHY");
    y=string_to_ints("LFDYTEERCLWQFFSRTWDREENIEGVLGQVARLLTGQTPQERLFYADALAMANDVRERFPWASQINHEEIHFLIDGLKSRLVDTVIQSSTNRELNHHLY");
    cherries[3] = {x,y};
    
    x=string_to_ints("LYEYVQERCLWQFFSRSWDREENIEGVLNQVVLLWSGKTPMERLFYADALPIVSDVKSRFEWASKIPAEEVSFLIDGLKTRLTETVITRSTNRELSHHLY");
    y=string_to_ints("LYQYIEERCLWQFFSRTWDREENIEGVLNQFGRLMTGETPMDRLFYADALPIANDCRERFDWAATITKDDIAELIASIKGQLVENTITRSTNRELSHHLY");
    cherries[4] = {x,y};

    std::vector<double> expected_lengths = {1.3577776670452830, 0.6334133023605781, 0.7664300958562997, 2.1867145105530996, 0.6967546325966361};

    test_lg_ble(cherries, expected_lengths);
}

void test_lg_ble3() {    
    std::vector<std::pair<std::vector<int>, std::vector<int> > > cherries(8);
    
    std::vector<int> x=string_to_ints("VYQAKLAEQAERYDEMVESMKKVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEENKGGEDKLKMIREYRQMVETELKLICCDILDVLDKHLIPAAESKVFYYKMKGDYHRYLAEFATGNDRKEAAENSLVAYKAASDIAMTELPPTHPIRLGLALNFSVFYYEILNSPDRACRLAKAAFDDAIAELDTLSEESYKDSTLIMQLLRDNLTLWTSD");
    std::vector<int> y=string_to_ints("VYLAKLAEQAERYEEMVENMKTVELSVEERNLLSVAYKNVIGARRASWRIVSSIEQKEESKESEHQVELICSYRSKIETELTKISDDILSVLDSHLIPSAESKVFYYKMKGDYHRYLAEFSSGDAREKATNASLEAYKTASEIATTELPPTHPIRLGLALNFSVFYYEIQNSPDKACHLAKQAFDDAIAELDTLSEESYKDSTLIMQLLRDNLTLWTSD");
    cherries[0] = {x,y};

    x=string_to_ints("VYLAKLAEQAERYEGMVENMKSVELTVEERNLLSVAYKNVIGARRASWRIVSSIEQKEESKGNTAQVELIKEYRQKIEQELDTICQDILTVLEKHLIPNAESKVFYYKMKGDYYRYLAEFAVGEKRQHSADQSLEGYKAASEIATAELAPTHPIRLGLALNFSVFYYEILNSPDRACYLAKQAFDEAISELDSLSEESYKDSTLIMQLLRDNLTLWTSD");
    y=string_to_ints("VYLAKLAEQAERYEEMVENMKKVKLSVEERNLLSVAYKNIIGARRASWRIISSIEQKEESRGNTRQAALIKEYRKKIEDELSDICHDVLSVLEKHLIPAAESKVFYYKMKGDYYRYLAEFTVGEVCKEAADSSLEAYKAASDIAVAELPPTDPMRLGLALNFSVFYYEILDSPESACHLAKQVFDEAISELDSLSEESYKDSTLIMQLLRDNLTLWTSD");
    cherries[1] = {x,y};
    
    x=string_to_ints("VYIAKLAEQAERYEEMVDSMKNVELTIEERNLLSVGYKNVIGARRASWRILSSIEQKEESKGNDVNAKRIKEYRHKVETELSNICIDVMRVIDEHLIPSAESTVFYYKMKGDYYRYLAEFKTGNEKKEAGDQSMKAYESATTAAEAELPPTHPIRLGLALNFSVFYYEILNSPERACHLAKQAFDEAISELDTLNEESYKDSTLIMQLLRDNLTLWTSD");
    y=string_to_ints("VYMAKLADRAESDEEMVEFMEKVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEESRGNEEHVNSIREYRSKIENELSKICDGILKLLDSKLIPSADSKVFYLKMKGDYHRYLAEFKTGAERKEAAESTLTAYKAAQDIASAELAPTHPIRLGLALNFSVFYYEILNSPDRACNLAKQAFDEAIAELDTLGEESYKDSTLIMQLLRDNLTLWTSD");
    cherries[2] = {x,y};
    
    x=string_to_ints("VYLAKLAEQAERYEEMIEFMEKVELTVEERNLLSVAYKNVIGARRASWRIISSIEQKEESRGNEDHVNTIKEYRSKIEADLSKICDGILSLLESNLIPSAESKVFHLKMKGDYHRYLAEFKTGTERKEAAENTLLAYKSAQDIALAELAPTHPIRLGLALNFSVFYYEILNSPDRACNLAKQAFDEAISELDTLGEESYKDSTLIMQLLRDNLTLWTSD");
    y=string_to_ints("VYMAKLAEQAERYEEMVQFMEQLELTVEERNLLSVAYKNVIGSLRAAWRIVSSIEQKEESRKNDEHVSLVKDYRSKVESELSSVCSGILKLLDSHLIPSAESKVFYLKMKGDYHRYMAEFKSGDERKTAAEDTMLAYKAAQDIAAADMAPTHPIRLGLALNFSVFYYEILNSSDKACNMAKQAFEEAIAELDTLGEESYKDSTLIMQLLRDNLTLWTSD");
    cherries[3] = {x,y};
    
    x=string_to_ints("VYLSKLAEQSERYEEMVQYMKQVELSVEERNLISVAYKNVVGSRRASWRIISSLEQKEQAKGNTQRVELIKTYRAKIEQELSQKCDDVLKIITEFLLKNSESKVFFKKMEGDYYRYYAEFTVDEKRKEVADKSLAAYQEATDTA-ASLVPTHPIRLGLALNFSVFYYQIMNDADKACQLAKEAFDEAIQKLDEVPEESYKESTLIMQLLRDNLTLWTSD");
    y=string_to_ints("VYTAKLAEQSERYDEMVQCMKQVELSIEERNLLSVAYKNVIGAKRASWRIISSLEQKEQAKGNDKHVEIIKGYRAKIEKELSTCCDDVLKVIQENLLPKAESKVFFKKMEGDYYRYFAEFTVDEKRKEVADKSLAAYTEATEISNAELAPTHPIRLGLALNFSVFYFEIMNDADKACQLAKQAFDDAIAKLDEVPENMYKDSTLIMQLLRDNLTLWTSD");
    cherries[4] = {x,y};
    
    x=string_to_ints("---AKLSEQAERYDDMAASMKAVELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTEG--NDKRQQMAREYREKVETELQDICKDVLDLLDRFLVPNAESKVFYLKMKGDYYRYLSEVASGDSKQETVASSQQAYQEAFEISKSEMQPTHPIRLGLALNFSVFYYEILNSPEKACSLAKSAFDEAIRELDTLNEESYKDSTLIMQLLRDNLTLWTSE");
    y=string_to_ints("IQKAKLAEQAERYDDMATCMKAVELSNEERNLLSVAYKNVVGGRRSAWRVISSIEQKTDT--SDKKLQLIKDYREKVESELRSICTTVLELLDKYLIANAESKVFYLKMKGDYFRYLAEVACGDDRKQTIDNSQGAYQEAFDISKKEMQPTHPIRLGLALNFSVFYYEILNNPELACTLAKTAFDEAIAELDTLNEDSYKDSTLIMQLLRDNLTLWTSD");
    cherries[5] = {x,y};
    
    x=string_to_ints("VQKAKLAEQSERYDDMAQAMKSVELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTEA--SARKQQLAREYRERVEKELREICYEVLGLLDKYLIPKAESKVFYLKMKGDYYRYLAEVATGDARNTVVDDSQTAYQDAFDISKGKMQPTHPIRLGLALNFSVFYYEILNSPDKACQLAKQAFDDAIAELDTLNEDSYKDSTLIMQLLRDNLTLWTSD");
    y=string_to_ints("VQRAKLAEQAERYDDMAAAMKKVELSNEERNLLSVAYKNVVGARRSSWRVISSIEQKTEG--SEKKQQLAKEYRVKVEQELNDICQDVLKLLDEFLIVKAESKAFYLKMKGDYYRYLAEVAS-EDRAAVVEKSQKAYQEALDIAKDKMQPTHPIRLGLALNFSVFYYEILNTPEHACQLAKQAFDDAIAELDTLNEDSYKDSTLIMQLLRDNLTLWTSD");
    cherries[6] = {x,y};

    x=string_to_ints("LQRARLAEQAERYDDMASAMKAVPLSNEDRNLLSVAYKNVVGARRSSWRVISSIEQKTMADGNEKKLEKVKAYREKIEKELETVCNDVLALLDKFLIKNCESKVFYLKMKGDYYRYLAEVASGEKKNSVVEASEAAYKEAFEISKEHMQPTHPIRLGLALNFSVFYYEIQNAPEQACLLAKQAFDDAIAELDTLNEDSYKDSTLIMQLLRDNLTLWTSD");
    y=string_to_ints("IQKAKLAEQAERYEDMAAFMKGAELSCEERNLLSVAYKNVVGGQRAAWRVLSSIEQKSNEEGSEEKGPEVREYREKVETELQGVCDTVLGLLDSHLIKEAESRVFYLKMKGDYYRYLAEVATGDDKKRIIDSARSAYQEAMDISKKEMPPTNPIRLGLALNFSVFHYEIANSPEEAISLAKTTFDEAMADLHTLSEDSYKDSTLIMQLLRDNLTLWTAD");
    cherries[7] = {x,y};

    std::vector<double> expected_lengths = {0.3932998257449988, 0.2954919802742289, 0.5234820680665935, 0.3250411783016518, 0.3250411783016518, 0.3932998257449988, 0.2954919802742289, 0.7664300958562997};

    test_lg_ble(cherries, expected_lengths);
}
int main() {
    test_lg_ble1();
    test_lg_ble2();
    test_lg_ble3();
    test_get_site_rates1();
    test_get_site_rates2();
    test_get_site_rates3();
    test_get_site_rates4();
    test_get_site_rates5();
    test_get_site_rates6();
    test_branch_lengths1();
    test_branch_lengths2();
    test_branch_lengths3();

    return 0;
}
