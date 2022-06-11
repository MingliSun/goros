//
// Created by sun on 2022/1/19.
//
/*
 * adapted from TASO
 */
#include"layer.h"
#include <vector>

TensorHandle squeeze(Graph* graph,  TensorHandle input, int oc) {
    auto conv=conv2d(graph,input,oc,1,1,PD_MODE_SAME,"");
    return graph->relu(conv);
}

TensorHandle fit(Graph* graph, TensorHandle current, TensorHandle input) {
    if (input->shape[2] == current->shape[2]) {
        return squeeze(graph, input, current->shape[1].get_upper_bound());
    }
    auto conv = conv2d(graph,input,current->shape[1].get_upper_bound(),3,2,PD_MODE_SAME,"");
    return graph->relu(conv);
}

TensorHandle separable_conv(Graph* graph, TensorHandle input, int oc,
                            int kernel,int stride,
                            PaddingMode padding) {
    assert(input->shape[1].get_upper_bound() % oc == 0);
    auto w1 = graph->new_weight({ oc, input->shape[1].get_upper_bound() / oc, kernel, kernel },DT_FLOAT,"w1");
    auto t= graph->conv2d_group(input,w1,stride,stride,padding);
    auto w2 = graph->new_weight({ oc, t->shape[1].get_upper_bound(), 1, 1 },DT_FLOAT,"w2");
    return graph->conv2d(t, w2, 1, 1, PD_MODE_SAME);
}

TensorHandle normal_cell(Graph* graph, TensorHandle prev, TensorHandle cur, int oc) {
    cur = squeeze(graph, cur, oc);
    prev = fit(graph, cur, prev);
    std::vector<TensorHandle> ts;
    ts.push_back(separable_conv(graph, cur, oc, 3, 1, PD_MODE_SAME));
    ts.push_back(cur);
    ts.push_back(separable_conv(graph, prev, oc, 3, 1, PD_MODE_SAME));
    ts.push_back(separable_conv(graph, cur, oc, 3, 1, PD_MODE_SAME));
    ts.push_back(graph->avg_pool2d(cur, 3, 3, 1, 1, PD_MODE_SAME));
    ts.push_back(prev);
    ts.push_back(graph->avg_pool2d(prev, 3, 3, 1, 1, PD_MODE_SAME));
    ts.push_back(graph->avg_pool2d(prev, 3, 3, 1, 1, PD_MODE_SAME));
    ts.push_back(separable_conv(graph, prev, oc, 3, 1, PD_MODE_SAME));
    ts.push_back(separable_conv(graph, prev, oc, 3, 1, PD_MODE_SAME));
    assert(ts.size() == 10);
    std::vector<TensorHandle> outputs;
    outputs.reserve(5);
    for (int i = 0; i < 5; i++) {
        outputs.push_back(graph->add(ts[2 * i], ts[2 * i + 1]));
    }
    return graph->concat(outputs,1);
}

TensorHandle reduction_cell(Graph* graph, TensorHandle prev, TensorHandle cur, int oc) {
    cur = squeeze(graph, cur, oc);
    prev = fit(graph, cur, prev);
    std::vector<TensorHandle> ts;
    std::vector<TensorHandle> outputs;
    ts.push_back(separable_conv(graph, prev, oc, 7, 2, PD_MODE_SAME));
    ts.push_back(separable_conv(graph, cur, oc, 5, 2, PD_MODE_SAME));
    outputs.push_back(graph->add(ts[0], ts[1]));
    ts.push_back(graph->max_pool2d(cur, 3, 3, 2, 2, PD_MODE_SAME));
    ts.push_back(separable_conv(graph, prev, oc, 7, 2, PD_MODE_SAME));
    outputs.push_back(graph->add(ts[2], ts[3]));
    ts.push_back(graph->avg_pool2d(cur, 3, 3, 2, 2, PD_MODE_SAME));
    ts.push_back(separable_conv(graph, prev, oc, 5, 2, PD_MODE_SAME));
    outputs.push_back(graph->add(ts[4], ts[5]));
    ts.push_back(graph->max_pool2d(cur, 3, 3, 2, 2, PD_MODE_SAME));
    ts.push_back(separable_conv(graph, outputs[0], oc, 3, 1, PD_MODE_SAME));
    outputs.push_back(graph->add( ts[6], ts[7]));
    ts.push_back(graph->avg_pool2d(outputs[0], 3, 3, 1, 1, PD_MODE_SAME));
    ts.push_back(outputs[1]);
    outputs.push_back(graph->add(ts[8], ts[9]));
    return graph->concat({outputs[2],outputs[3],outputs[4]},1);
}

Graph* nasnet_a() {
    auto graph = new Graph();
    auto inp = graph->new_input({ 1, 3, 224, 224 },DT_FLOAT,"data");
    inp = conv2d(graph,inp,64,7,2,PD_MODE_SAME,"");
    inp = graph->max_pool2d(inp, 3, 3, 2, 2, PD_MODE_SAME);
    int oc = 128;
    for (int i = 0; i < 3; i++) {
        auto prev = inp;
        auto cur = inp;
        for (int j = 0; j < 5; j++) {
            auto t = normal_cell(graph, prev, cur, oc);
            prev = cur;
            cur = t;
        }
        oc *= 2;
        inp = reduction_cell(graph, prev, cur, oc);
    }
    graph->function({inp});
    return graph;
}