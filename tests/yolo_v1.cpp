//
// Created by sun on 2021/12/29.
//

#include"layer.h"

Graph* yolo_v1(){
    int batch = 1;
    auto * graph = new Graph();
    TensorHandle data = graph->new_input({batch,1,448,448},DT_FLOAT,"data");
    auto conv1 = conv2d(graph,data,64,7,2,PD_MODE_SAME,"conv1");
    auto act1 = graph->relu(conv1);
    auto pool1 = graph->max_pool2d(act1,2,2,2,2,PD_MODE_SAME);

    auto conv2 = conv2d(graph,pool1,192,3,1,PD_MODE_SAME,"conv2");
    auto act2 = graph->relu(conv2);
    auto pool2 = graph->max_pool2d(act2,2,2,2,2,PD_MODE_SAME);

    auto conv3 = conv2d(graph,pool2,128,1,1,PD_MODE_SAME,"conv3_1");
    auto act3 = graph->relu(conv3);
    conv3 = conv2d(graph,act3,256,3,1,PD_MODE_SAME,"conv3_2");
    act3 = graph->relu(conv3);
    conv3 = conv2d(graph,act3,256,1,1,PD_MODE_SAME,"conv3_3");
    act3 = graph->relu(conv3);
    conv3 = conv2d(graph,act3,512,3,1,PD_MODE_SAME,"conv3_4");
    act3 = graph->relu(conv3);
    auto pool3 = graph->max_pool2d(act3,2,2,2,2,PD_MODE_SAME);
    //==================
    auto act4 = pool3;
    TensorHandle conv4;
    for(int i=1;i<=4;i++){
        conv4 = conv2d(graph,act4,256,1,1,PD_MODE_SAME,"conv4_"+::to_string(i)+"_1");
        act4 = graph->relu(conv4);
        conv4 = conv2d(graph,act4,512,3,1,PD_MODE_SAME,"conv4_2"+::to_string(i)+"_2");
        act4 = graph->relu(conv4);
    }
    conv4 = conv2d(graph,act4,512,1,1,PD_MODE_SAME,"conv4_5");
    act4 = graph->relu(conv4);
    conv4 = conv2d(graph,act4,1024,3,1,PD_MODE_SAME,"conv4_6");
    act4 = graph->relu(conv4);
    auto pool4 = graph->max_pool2d(act4,2,2,2,2,PD_MODE_SAME);

    auto act5 = pool4;
    TensorHandle conv5;
    for(int i=1;i<=2;i++){
        conv5 = conv2d(graph,act5,512,1,1,PD_MODE_SAME,"conv5"+::to_string(i)+"_1");
        act5 = graph->relu(conv5);
        conv5 = conv2d(graph,act5,1024,3,1,PD_MODE_SAME,"conv5"+::to_string(i)+"_1");
        act5 = graph->relu(conv5);
    }
    conv5 = conv2d(graph,act5,1024,3,1,PD_MODE_SAME,"conv5_3");
    act5 = graph->relu(conv5);
    conv5 = conv2d(graph,act5,1024,3,2,PD_MODE_SAME,"conv5_4");
    act5 = graph->relu(conv5);

    auto conv6 = conv2d(graph,act5,1024,3,1,PD_MODE_SAME,"conv6_1");
    auto act6 = graph->relu(conv6);
    conv6 = conv2d(graph,act6,1024,3,1,PD_MODE_SAME,"conv6_2");
    act6 = graph->relu(conv6);
    auto flat = graph->batch_flatten(act6);
    auto fc1 = dense_add_bias(graph,flat,4096,"fc1");
    auto fc2 = dense_add_bias(graph,fc1,1470,"fc2");
    fc2 = graph->reshape(fc2,{batch,7,7,30});
    graph->function({fc2});
    return graph;
}

void test_yolo_v1(){
    auto graph = yolo_v1();
    string name = __FUNCTION__ ;
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(name+".dot");
    graph->codegen_te(name+".py");
}