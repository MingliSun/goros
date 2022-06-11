//
// Created by sun on 2021/12/9.
//

#include"Ops.h"
#include<iostream>

void unet(){
    auto * graph = new Graph();
    TensorHandle inputs = graph->new_input({1,1,512,512},DT_FLOAT,"inputs");
    TensorHandle conv1_weight_0 = graph->new_weight({64,1,3,3},DT_FLOAT,"conv1_weight_0");
    TensorHandle conv1 = graph->conv2d(inputs,conv1_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu1 = graph->relu(conv1);
    TensorHandle conv1_weight_1 = graph->new_weight({64,64,3,3},DT_FLOAT,"conv1_weight_1");
                  conv1 = graph->conv2d(relu1,conv1_weight_1,1,1,PD_MODE_SAME);
                  relu1 = graph->relu(conv1);
    TensorHandle pool1 = graph->max_pool2d(relu1, 2, 2, 2, 2, PD_MODE_VALID);
    TensorHandle conv2_weight_0 = graph->new_weight({128,64,3,3},DT_FLOAT,"conv2_weight_0");
    TensorHandle conv2 = graph->conv2d(pool1,conv2_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu2 = graph->relu(conv2);
    TensorHandle conv2_weight_1 = graph->new_weight({128,128,3,3},DT_FLOAT,"conv2_weight_1");
                 conv2 = graph->conv2d(relu2,conv2_weight_1,1,1,PD_MODE_SAME);
                 relu2 = graph->relu(conv2);
    TensorHandle pool2 = graph->max_pool2d(relu2, 2, 2, 2, 2, PD_MODE_VALID);
    TensorHandle conv3_weight_0 = graph->new_weight({256,128,3,3},DT_FLOAT,"conv3_weight_0");
    TensorHandle conv3 = graph->conv2d(pool2,conv3_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu3=  graph->relu(conv3);
    TensorHandle conv3_weight_1 = graph->new_weight({256,256,3,3},DT_FLOAT,"conv3_weight_1");
                 conv3 = graph->conv2d(relu3,conv3_weight_1,1,1,PD_MODE_SAME);
                 relu3 = graph->relu(conv3);
    TensorHandle pool3 = graph->max_pool2d(relu3, 2, 2, 2, 2, PD_MODE_VALID);
    TensorHandle conv4_weight_0 = graph->new_weight({512,256,3,3},DT_FLOAT,"conv4_weight_0");
    TensorHandle conv4 = graph->conv2d(pool3,conv4_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu4=  graph->relu(conv4);
    TensorHandle conv4_weight_1 = graph->new_weight({512,512,3,3},DT_FLOAT,"conv4_weight_1");
                conv4 = graph->conv2d(relu4,conv4_weight_1,1,1,PD_MODE_SAME);
                relu4 = graph->relu(conv4);
    TensorHandle drop4 = graph->dropout(relu4,0.5);
    TensorHandle pool4 = graph->max_pool2d(drop4, 2, 2, 2, 2, PD_MODE_VALID);

    TensorHandle conv5_weight_0 = graph->new_weight({1024,512,3,3},DT_FLOAT,"conv5_weight_0");
    TensorHandle conv5 = graph->conv2d(pool4,conv5_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu5=  graph->relu(conv5);
    TensorHandle conv5_weight_1 = graph->new_weight({1024,1024,3,3},DT_FLOAT,"conv5_weight_1");
                conv5 = graph->conv2d(relu5,conv5_weight_1,1,1,PD_MODE_SAME);
                relu5 = graph->relu(conv5);
    TensorHandle drop5 = graph->dropout(relu5,0.5);



    TensorHandle up6 = graph->upsampling(drop5, 2, 2);////upSampling
    TensorHandle up6_weight_0 = graph->new_weight({512,1024,2,2},DT_FLOAT,"up6_weight_0");
                 up6 = graph->conv2d(up6,up6_weight_0,1,1,PD_MODE_SAME);
    TensorHandle up6_relu = graph->relu(up6);
    TensorHandle merge6 = graph->concat({drop4,up6_relu},1);
    TensorHandle conv6_weight_0 = graph->new_weight({512,1024,3,3},DT_FLOAT,"conv6_weight_0");
    TensorHandle conv6 = graph->conv2d(merge6,conv6_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu6 = graph->relu(conv6);
    TensorHandle conv6_weight_1 = graph->new_weight({512,512,2,2,},DT_FLOAT,"conv6_weight_1");
                 conv6 = graph->conv2d(relu6,conv6_weight_1,1,1,PD_MODE_SAME);
                 relu6 = graph->relu(conv6);

    TensorHandle up7 = graph->upsampling(relu6, 2, 2);////upSampling
    TensorHandle up7_weight_0 = graph->new_weight({256,512,2,2},DT_FLOAT,"up7_weight_0");
                up7 = graph->conv2d(up7,up7_weight_0,1,1,PD_MODE_SAME);
    TensorHandle up7_relu = graph->relu(up7);
    TensorHandle merge7 = graph->concat({relu3,up7_relu},1);
    TensorHandle conv7_weight_0 = graph->new_weight({256,512,3,3},DT_FLOAT,"conv7_weight_0");
    TensorHandle conv7 = graph->conv2d(merge7,conv7_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu7 = graph->relu(conv7);
    TensorHandle conv7_weight_1 = graph->new_weight({256,256,3,3,},DT_FLOAT,"conv7_weight_1");
    conv7 = graph->conv2d(relu7,conv7_weight_1,1,1,PD_MODE_SAME);
    relu7 = graph->relu(conv7);

    TensorHandle up8 = graph->upsampling(relu7, 2, 2);////upSampling
    TensorHandle up8_weight_0 = graph->new_weight({128,256,2,2},DT_FLOAT,"up8_weight_0");
                 up8 = graph->conv2d(up8,up8_weight_0,1,1,PD_MODE_SAME);
    TensorHandle up8_relu = graph->relu(up8);
    TensorHandle merge8 = graph->concat({relu2,up8_relu},1);
    TensorHandle conv8_weight_0 = graph->new_weight({128,256,3,3},DT_FLOAT,"conv8_weight_0");
    TensorHandle conv8 = graph->conv2d(merge8,conv8_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu8 = graph->relu(conv8);
    TensorHandle conv8_weight_1 = graph->new_weight({128,128,3,3,},DT_FLOAT,"conv8_weight_1");
    conv8 = graph->conv2d(relu8,conv8_weight_1,1,1,PD_MODE_SAME);
    relu8 = graph->relu(conv8);

    TensorHandle up9 = graph->upsampling(relu8, 2, 2);////upSampling
    TensorHandle up9_weight_0 = graph->new_weight({64,128,2,2},DT_FLOAT,"up9_weight_0");
    up9 = graph->conv2d(up9,up9_weight_0,1,1,PD_MODE_SAME);
    TensorHandle up9_relu = graph->relu(up9);
    TensorHandle merge9 = graph->concat({relu1,up9_relu},1);
    TensorHandle conv9_weight_0 = graph->new_weight({64,128,3,3},DT_FLOAT,"conv9_weight_0");
    TensorHandle conv9 = graph->conv2d(merge9,conv9_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu9 = graph->relu(conv9);
    TensorHandle conv9_weight_1 = graph->new_weight({64,64,3,3,},DT_FLOAT,"conv9_weight_1");
     conv9 = graph->conv2d(relu9,conv9_weight_1,1,1,PD_MODE_SAME);
     relu9 = graph->relu(conv9);

    TensorHandle conv9_weight_2 = graph->new_weight({2,64,3,3},DT_FLOAT,"conv9_weight_2");
    conv9 = graph->conv2d(relu9,conv9_weight_2,1,1,PD_MODE_SAME);
    relu9 = graph->relu(conv9);
    TensorHandle conv10_weight_0 = graph->new_weight({1,2,1,1},DT_FLOAT,"conv10_weight_0");
    TensorHandle conv10 = graph->conv2d(relu9,conv10_weight_0,1,1,PD_MODE_SAME);
    TensorHandle sigmoid10 = graph->sigmoid(conv10);

    cout<<graph->to_string()<<endl;
    cout<<"=============================================================================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}


void test_unet_1(){
    auto * graph = new Graph();
    //// relu_1
    TensorHandle inputs = graph->new_input({1,1,512,512},DT_FLOAT,"inputs");
    TensorHandle conv1_weight_0 = graph->new_weight({64,1,3,3},DT_FLOAT,"conv1_weight_0");
    TensorHandle conv1 = graph->conv2d(inputs,conv1_weight_0,1,1,PD_MODE_SAME);
    TensorHandle relu1 = graph->relu(conv1);//[1,64,512,512]


    //// Assuming relu_8 is input
    TensorHandle relu8 = graph->new_input({1,128,256,256},DT_FLOAT,"relu8");
    TensorHandle up9 = graph->upsampling(relu8, 2, 2);////upSampling
    TensorHandle up9_weight_0 = graph->new_weight({64,128,2,2},DT_FLOAT,"up9_weight_0");
    up9 = graph->conv2d(up9,up9_weight_0,1,1,PD_MODE_SAME);
    TensorHandle up9_relu = graph->relu(up9);
    TensorHandle merge9 = graph->concat({relu1,up9_relu},1);

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}

void test_unet_2(){
    auto graph = new Graph;

    ////relu4 as input
    TensorHandle relu4 = graph->new_input({1,512,64,64},DT_FLOAT,"relu4");
    TensorHandle conv4_weight_1 = graph->new_weight({512,512,3,3},DT_FLOAT,"conv4_weight_1");
    TensorHandle conv4 = graph->conv2d(relu4,conv4_weight_1,1,1,PD_MODE_SAME);
    relu4 = graph->relu(conv4);
    TensorHandle drop4 = graph->dropout(relu4,0.5);

    //// relu5 as input
    TensorHandle relu5 = graph->new_input({1,1024,32,32},DT_FLOAT,"relu5");
    TensorHandle drop5 = graph->dropout(relu5,0.5);
    TensorHandle up6 = graph->upsampling(drop5, 2, 2);////upSampling
    TensorHandle up6_weight_0 = graph->new_weight({512,1024,2,2},DT_FLOAT,"up6_weight_0");
    up6 = graph->conv2d(up6,up6_weight_0,1,1,PD_MODE_SAME);
    TensorHandle up6_relu = graph->relu(up6);
    TensorHandle merge6 = graph->concat({drop4,up6_relu},1);

    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}