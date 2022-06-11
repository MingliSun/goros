//
// Created by sun on 2022/1/19.
//

/*
 * Adapted from https://github.com/qingzhouzhen/incubator-mxnet/blob/nasnet/python/mxnet/gluon/model_zoo/vision/nasnet.py
 * todo Parameter is never used?
 */
#include"layer.h"


TensorHandle separable_conv2d(Graph* graph,TensorHandle input,
                                  int depthwise_channels,
                                  int pointwise_channels,
                                  int kernel=3 ,
                                  int stride=1){
    int ic = input->shape[1].get_upper_bound();
    auto weight = graph->new_weight({depthwise_channels,ic/depthwise_channels,kernel,kernel},DT_FLOAT,"_conv1_weight");
    auto conv1 = graph->conv2d_group(input,weight,stride,stride,PD_MODE_SAME);

    auto conv2 = conv2d(graph,conv1,pointwise_channels,1,1,PD_MODE_SAME,"_conv2");
    TensorHandle bias = graph->new_weight(pointwise_channels,1,DT_FLOAT,"_bias");
    return graph->bias_add(conv2,bias);
}

TensorHandle branch_separable(Graph* graph,TensorHandle input,
                              int ic,
                              int oc,
                              int kernel,
                              int stride){
    auto x = graph->relu(input);
    x = separable_conv2d(graph,x,ic,ic,kernel,stride);
    x = graph->relu(x);
    x = separable_conv2d(graph,x,ic,oc,kernel,1);
    return x;
}

TensorHandle branch_separable_stem(Graph* graph,TensorHandle input,
                              int ic,
                              int oc,
                              int kernel,
                              int stride){
    auto x = graph->relu(input);
    x = separable_conv2d(graph,x,ic,oc,kernel,stride);
    x = graph->relu(x);
    x = separable_conv2d(graph,x,oc,oc,kernel,1);
    return x;
}



TensorHandle normal_cell(Graph* graph, TensorHandle prev, TensorHandle cur, int ic_left,
                         int oc_left,
                         int ic_right,
                         int oc_right) {
    auto x_left = graph->relu(prev);
    x_left = conv2d(graph,x_left,oc_left,1,1,PD_MODE_SAME,"");
    auto x_right = graph->relu(cur);
    x_right = conv2d(graph,x_right,oc_right,1,1,PD_MODE_SAME,"");

    std::vector<TensorHandle> ts;
    ts.push_back(branch_separable(graph, x_right, oc_right,oc_right, 5,1));
    ts.push_back(branch_separable(graph, x_left, oc_left,oc_left, 3,1));

    ts.push_back(branch_separable(graph, x_left, oc_left,oc_left, 5,1));
    ts.push_back(branch_separable(graph, x_left, oc_left,oc_left, 3,1));

    ts.push_back(graph->avg_pool2d(x_right,3,3,1,1,PD_MODE_SAME));
    ts.push_back(x_left);

    ts.push_back(graph->avg_pool2d(x_left,3,3,1,1,PD_MODE_SAME));
    ts.push_back(graph->avg_pool2d(x_left,3,3,1,1,PD_MODE_SAME));

    ts.push_back(branch_separable(graph, x_right, oc_right,oc_right, 3,1));
    ts.push_back(x_right);

    assert(ts.size() == 10);
    std::vector<TensorHandle> outputs;
    outputs.reserve(5);
    for (int i = 0; i < 5; i++) {
        outputs.push_back(graph->add(ts[2 * i], ts[2 * i + 1]));
    }
    return graph->concat(outputs,1);
}

TensorHandle cell_stem_0(Graph* graph, TensorHandle x){
    auto x1 = graph->relu(x);
    x1 = conv2d(graph,x1,42,1,1,PD_MODE_SAME,"");
    std::vector<TensorHandle> ts;
    std::vector<TensorHandle> outputs;
    outputs.reserve(5);
    ts.push_back(branch_separable(graph, x1, 42,42, 5,2));
    ts.push_back(branch_separable_stem(graph, x, 96,42, 7,2));
    outputs.push_back(graph->add(ts[0],ts[1]));
    ts.push_back(graph->max_pool2d(x1,3,3,2,2,PD_MODE_SAME));
    ts.push_back(branch_separable_stem(graph, x, 96,42, 7,2));
    outputs.push_back(graph->add(ts[2],ts[3]));
    ts.push_back(graph->avg_pool2d(x1,3,3,2,2,PD_MODE_SAME));
    ts.push_back(branch_separable_stem(graph, x, 96,42, 5,2));
    outputs.push_back(graph->add(ts[4],ts[5]));

    ts.push_back(graph->avg_pool2d(outputs[0],3,3,1,1,PD_MODE_SAME));
    ts.push_back(outputs[1]);
    outputs.push_back(graph->add(ts[6],ts[7]));

    ts.push_back(branch_separable(graph, outputs[0], 42,42, 3,1));
    ts.push_back(graph->max_pool2d(x1,3,3,2,2,PD_MODE_SAME));
    outputs.push_back(graph->add(ts[8],ts[9]));

    return graph->concat(outputs,1);
}
TensorHandle cell_stem_1(Graph* graph, TensorHandle x_conv0,TensorHandle x_stem_0){
     auto x_left = graph->relu(x_stem_0);
     x_left = conv2d(graph,x_left,84,1,1,PD_MODE_SAME,"");
     auto x_relu = graph->relu(x_conv0);
     auto x_path1 = graph->avg_pool2d(x_relu,1,1,2,2,PD_MODE_SAME);
     x_path1 = conv2d(graph,x_path1,42,1,1,PD_MODE_SAME,"");
     auto x_path2 = graph->avg_pool2d(x_relu,1,1,2,2,PD_MODE_SAME);
     x_path2 = conv2d(graph,x_path2,42,1,1,PD_MODE_SAME,"");
     auto x_right = graph->concat({x_path1,x_path2},1);

    std::vector<TensorHandle> ts;
    std::vector<TensorHandle> outputs;
    outputs.reserve(5);
    ts.push_back(branch_separable(graph, x_left, 84,84, 5,2));
    ts.push_back(branch_separable(graph, x_right, 84,84, 7,2));
    outputs.push_back(graph->add(ts[0],ts[1]));

    ts.push_back(graph->max_pool2d(x_left,3,3,2,2,PD_MODE_SAME));
    ts.push_back(branch_separable(graph, x_right, 84,84, 7,2));
    outputs.push_back(graph->add(ts[2],ts[3]));

    ts.push_back(graph->avg_pool2d(x_left,3,3,2,2,PD_MODE_SAME));
    ts.push_back(branch_separable(graph, x_right, 84,84, 5,2));
    outputs.push_back(graph->add(ts[4],ts[5]));

    ts.push_back(graph->avg_pool2d(outputs[0],3,3,1,1,PD_MODE_SAME));
    ts.push_back(outputs[1]);
    outputs.push_back(graph->add(ts[6],ts[7]));

    ts.push_back(branch_separable(graph, outputs[0], 84,84, 3,1));
    ts.push_back(graph->max_pool2d(x_left,3,3,2,2,PD_MODE_SAME));
    outputs.push_back(graph->add(ts[8],ts[9]));

    return graph->concat(outputs,1);
}

TensorHandle first_cell(Graph* graph, TensorHandle prev, TensorHandle cur, int ic_left,
                        int oc_left,
                        int ic_right,
                        int oc_right){
    auto x_relu = graph->relu(prev);
    auto x_path1 = graph->avg_pool2d(x_relu,1,1,2,2,PD_MODE_SAME);
    x_path1 = conv2d(graph,x_path1,oc_left,1,1,PD_MODE_SAME,"");
    auto x_path2 = graph->avg_pool2d(x_relu,1,1,2,2,PD_MODE_SAME);
    x_path2 = conv2d(graph,x_path2,oc_left,1,1,PD_MODE_SAME,"");
    auto x_left = graph->concat({x_path1,x_path2},1);

    auto x_right = graph->relu(cur);
    x_right = conv2d(graph,x_right,oc_right,1,1,PD_MODE_SAME,"");

    std::vector<TensorHandle> ts;
    std::vector<TensorHandle> outputs;
    outputs.reserve(5);
    ts.push_back(branch_separable(graph, x_right, oc_right,oc_right, 5,1));
    ts.push_back(branch_separable(graph, x_left, oc_right,oc_right, 3,1));
    outputs.push_back(graph->add(ts[0],ts[1]));

    ts.push_back(branch_separable(graph, x_right, oc_right,oc_right, 5,1));
    ts.push_back(branch_separable(graph, x_left, oc_right,oc_right, 3,1));
    outputs.push_back(graph->add(ts[2],ts[3]));

    ts.push_back(graph->avg_pool2d(x_right,3,3,1,1,PD_MODE_SAME));
    ts.push_back(x_left);
    outputs.push_back(graph->add(ts[4],ts[5]));


    ts.push_back(graph->avg_pool2d(x_left,3,3,1,1,PD_MODE_SAME));
    ts.push_back(graph->avg_pool2d(x_left,3,3,1,1,PD_MODE_SAME));
    outputs.push_back(graph->add(ts[6],ts[7]));

    ts.push_back(branch_separable(graph, x_right, oc_right,oc_right, 3,1));
    ts.push_back(x_right);
    outputs.push_back(graph->add(ts[8],ts[9]));

    return graph->concat(outputs,1);

}
TensorHandle reduction_cell0(Graph* graph, TensorHandle prev, TensorHandle cur, int ic_left,
                         int oc_left,
                         int ic_right,
                         int oc_right) {
    auto x_left = graph->relu(prev);
    x_left = conv2d(graph,x_left,oc_left,1,1,PD_MODE_SAME,"");
    auto x_right = graph->relu(cur);
    x_right = conv2d(graph,x_right,oc_right,1,1,PD_MODE_SAME,"");

    std::vector<TensorHandle> ts;
    std::vector<TensorHandle> outputs;
    outputs.reserve(5);
    ts.push_back(branch_separable(graph, x_right, oc_right,oc_right, 5,2));
    ts.push_back(branch_separable(graph, x_left, oc_right,oc_right, 7,2));
    outputs.push_back(graph->add(ts[0],ts[1]));

    ts.push_back(graph->max_pool2d(x_right,3,3,2,2,PD_MODE_SAME));
    ts.push_back(branch_separable(graph, x_left, oc_right,oc_right, 7,2));
    outputs.push_back(graph->add(ts[2],ts[3]));

    ts.push_back(graph->max_pool2d(x_right,3,3,2,2,PD_MODE_SAME));
    ts.push_back(branch_separable(graph, x_left, oc_right,oc_right, 5,2));
    outputs.push_back(graph->add(ts[4],ts[5]));


    ts.push_back(graph->avg_pool2d(outputs[0],3,3,1,1,PD_MODE_SAME));
    ts.push_back(outputs[1]);
    outputs.push_back(graph->add(ts[6],ts[7]));

    ts.push_back(branch_separable(graph, outputs[0], oc_right,oc_right, 3,1));
    ts.push_back(graph->max_pool2d(x_right,3,3,2,2,PD_MODE_SAME));
    outputs.push_back(graph->add(ts[8],ts[9]));

    return graph->concat(outputs,1);

}
TensorHandle reduction_cell1(Graph* graph, TensorHandle prev, TensorHandle cur, int ic_left,
                             int oc_left,
                             int ic_right,
                             int oc_right) {
    auto x_left = graph->relu(prev);
    x_left = conv2d(graph,x_left,oc_left,1,1,PD_MODE_SAME,"");
    auto x_right = graph->relu(cur);
    x_right = conv2d(graph,x_right,oc_right,1,1,PD_MODE_SAME,"");

    std::vector<TensorHandle> ts;
    std::vector<TensorHandle> outputs;
    outputs.reserve(5);
    ts.push_back(branch_separable(graph, x_right, oc_right,oc_right, 5,2));
    ts.push_back(branch_separable(graph, x_left, oc_right,oc_right, 7,2));
    outputs.push_back(graph->add(ts[0],ts[1]));

    ts.push_back(graph->max_pool2d(x_right,3,3,2,2,PD_MODE_SAME));
    ts.push_back(branch_separable(graph, x_left, oc_right,oc_right, 7,2));
    outputs.push_back(graph->add(ts[2],ts[3]));

    ts.push_back(graph->avg_pool2d(x_right,3,3,2,2,PD_MODE_SAME));
    ts.push_back(branch_separable(graph, x_left, oc_right,oc_right, 5,2));
    outputs.push_back(graph->add(ts[4],ts[5]));


    ts.push_back(graph->avg_pool2d(outputs[0],3,3,1,1,PD_MODE_SAME));
    ts.push_back(outputs[1]);
    outputs.push_back(graph->add(ts[6],ts[7]));

    ts.push_back(branch_separable(graph, outputs[0], oc_right,oc_right, 3,1));
    ts.push_back(graph->max_pool2d(x_right,3,3,2,2,PD_MODE_SAME));
    outputs.push_back(graph->add(ts[8],ts[9]));

    return graph->concat(outputs,1);

}

Graph* nasnet(int num_classes) {
    auto graph = new Graph();
    auto data = graph->new_input({1,3,224,224},DT_FLOAT,"data");
    auto x_conv0 = conv2d(graph,data,96,3,2,PD_MODE_VALID,"");
    auto x_stem_0 = cell_stem_0(graph, x_conv0);
    auto inp = cell_stem_1(graph,x_conv0,x_stem_0);
    int oc = 168;
    for (int i = 0; i < 3; i++) {
        auto cell0 = first_cell(graph,x_stem_0,inp,168,oc/2,336,oc);
        auto prev = inp;
        auto cur = cell0;
        for (int j = 0; j < 5; j++) {
            auto t = normal_cell(graph, prev, cur, 336,oc,1008,oc);
            prev = cur;
            cur = t;
        }
        oc*=2;
        if(i==0) inp = reduction_cell0(graph, prev, cur, 1008,oc,1008,oc);
        else if(i==2) inp = reduction_cell1(graph, prev, cur, 1008,oc,1008,oc);
        else inp = cur;
    }
    auto x =graph->relu(inp);
    x = graph->avg_pool2d(x,11,11,1,1,PD_MODE_VALID);
    x = graph->batch_flatten(x);
    x = graph->dropout(x,0.5);
    x = dense_add_bias(graph,x,num_classes,"");
    graph->function({x});
    return graph;
}