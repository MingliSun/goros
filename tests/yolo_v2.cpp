//
// Created by sun on 2021/12/29.
//

/*
 *  ignoring batch_norm
 */

#include"layer.h"

Graph* yolo_v2(){
    auto graph = new Graph;
    TensorHandle data = graph->new_input({1,3,416,416},DT_FLOAT,"data");
    auto conv1 = conv2d(graph,data,32,3,1,PD_MODE_SAME,"conv1");
    auto act1 = graph->leaky_relu(conv1);
    auto pool1 = graph->max_pool2d(act1,2,2,2,2,PD_MODE_SAME);

    auto conv2 = conv2d(graph,pool1,64,3,1,PD_MODE_SAME,"conv2");
    auto act2 = graph->leaky_relu(conv2);
    auto pool2 = graph->max_pool2d(act2,2,2,2,2,PD_MODE_SAME);

    auto conv3 = conv2d(graph,pool2,128,3,1,PD_MODE_SAME,"conv3_1");
    auto act3 = graph->leaky_relu(conv3);

    conv3 = conv2d(graph,act3,64,1,1,PD_MODE_SAME,"conv3_2");
    act3= graph->leaky_relu(conv3);

    conv3 = conv2d(graph,act3,128,3,1,PD_MODE_SAME,"conv3_3");
    act3 = graph->leaky_relu(conv3);

    auto pool3 = graph->max_pool2d(act3,2,2,2,2,PD_MODE_SAME);

    auto conv4 = conv2d(graph,pool3,256,3,1,PD_MODE_SAME,"conv4_1");
    auto act4 = graph->leaky_relu(conv4);

    conv4 = conv2d(graph,act4,128,1,1,PD_MODE_SAME,"conv4_2");
    act4= graph->leaky_relu(conv4);

    conv4 = conv2d(graph,act4,256,3,1,PD_MODE_SAME,"conv4_3");
    act4 = graph->leaky_relu(conv4);

    auto pool4 = graph->max_pool2d(act4,2,2,2,2,PD_MODE_SAME);

    auto conv5 = conv2d(graph,pool4,512,3,1,PD_MODE_SAME,"conv5_1");
    auto act5 = graph->leaky_relu(conv5);

    conv5 = conv2d(graph,act5,256,1,1,PD_MODE_SAME,"conv5_2");
    act5= graph->leaky_relu(conv5);

    conv5 = conv2d(graph,act5,512,3,1,PD_MODE_SAME,"conv5_3");
    act5 = graph->leaky_relu(conv5);

    conv5 = conv2d(graph,act5,256,1,1,PD_MODE_SAME,"conv5_4");
    act5= graph->leaky_relu(conv5);

    conv5 = conv2d(graph,act5,512,3,1,PD_MODE_SAME,"conv5_5");
    act5 = graph->leaky_relu(conv5);// shortcut begin
    auto pool5 = graph->max_pool2d(act5,2,2,2,2,PD_MODE_SAME);

    auto conv6 = conv2d(graph,pool5,1024,3,1,PD_MODE_SAME,"conv6_1");
    auto act6 = graph->leaky_relu(conv6);

    conv6 = conv2d(graph,act6,512,1,1,PD_MODE_SAME,"conv6_2");
    act6= graph->leaky_relu(conv6);

    conv6 = conv2d(graph,act6,1024,3,1,PD_MODE_SAME,"conv6_3");
    act6 = graph->leaky_relu(conv6);

    conv6 = conv2d(graph,act6,512,1,1,PD_MODE_SAME,"conv6_4");
    act6= graph->leaky_relu(conv6);

    conv6 = conv2d(graph,act6,1024,3,1,PD_MODE_SAME,"conv6_5");
    act6 = graph->leaky_relu(conv6);

    auto conv7 = conv2d(graph,act6,1024,3,1,PD_MODE_SAME,"conv7_1");
    auto act7 = graph->leaky_relu(conv7);

    conv7 = conv2d(graph,act7,512,1,1,PD_MODE_SAME,"conv7_2");
    act7= graph->leaky_relu(conv7);

    //shortcut
    auto shortcut = conv2d(graph,act5,64,1,1,PD_MODE_SAME,"conv_shortcut");
    auto shortcut_act = graph->leaky_relu(shortcut);
    shortcut = graph->yolo_reorg(shortcut_act,2);

    auto concat = graph->concat({act7,shortcut},1);
    auto conv8 = conv2d(graph,concat,1024,3,1,PD_MODE_SAME,"conv8");
    auto act8 = graph->leaky_relu(conv8);

    auto conv = conv2d(graph,act8,425,1,1,PD_MODE_SAME,"conv_dec");
    auto net = bias_add(graph,conv,"conv_bias");
    net = graph->reshape(net,{-1,5,85,13,13});
    vector<TensorHandle> out;
    out.resize(4);
    graph->split(net,{2,4,5},2,out);
    out[0] = graph->sigmoid(out[0]);
    out[2] = graph->sigmoid(out[2]);
    out[3] = graph->softmax(out[3],2);
    net = graph->concat(out,2);
    net = graph->reshape(net,{-1,425,13,13});
    graph->function({net});
    return graph;
}