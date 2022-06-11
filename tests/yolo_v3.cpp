//
// Created by sun on 2021/12/31.
//

#include"layer.h"

/*
 * yolo_v3 ignoring batch_norm
 */

TensorHandle dbl(Graph* graph, TensorHandle input, int oc, int kernel, int stride, PaddingMode p, const string& name){
    auto conv = conv2d(graph,input,oc,kernel,stride,p,name);
    return graph->leaky_relu(conv);
}

TensorHandle dbl5(Graph* graph, TensorHandle input, int oc, const string& name){
    auto body = input;
    for(int i=0;i<5;i++){
        if(i%2==0) body = dbl(graph,body,oc,1,1,PD_MODE_SAME,name+::to_string(i+1));
        else body = dbl(graph,body,oc*2,3,1,PD_MODE_SAME,name+::to_string(i+1));
    }
    return body;
}

TensorHandle residual_unit(Graph* graph,TensorHandle input,int oc,const string& name){
    auto dbl1 = dbl(graph,input,oc/2,1,1,PD_MODE_SAME,name+"_dbl1");
    auto dbl2 = dbl(graph,dbl1,oc,3,1,PD_MODE_SAME,name+"_dl2");
    return graph->add(dbl2,input);
}

TensorHandle block_body(int n,Graph* graph,TensorHandle input,int oc,const string& name){
    auto body = dbl(graph,input,oc,3,2,PD_MODE_SAME,name+"_dbl"); //stride=2
    for(int i=0;i<n;i++){
        body = residual_unit(graph,body,oc,name+"_res"+::to_string(i+1));
    }
    return body;
}

Graph* yolo_v3(){
    auto graph = new Graph;
    int size = 608;
    auto data = graph->new_input({1,3,size,size},DT_FLOAT,"data");
    auto conv = dbl(graph,data,32,3,1,PD_MODE_SAME,"conv");
    auto res1 = block_body(1,graph,conv,64,"res1");
    auto res2 = block_body(2,graph,res1,128,"res2");
    auto res8 = block_body(8,graph,res2,256,"res8");
    auto res8_2 = block_body(8,graph,res8,512,"res8_2");
    auto res4 = block_body(4,graph,res8_2,1024,"res4");

    auto dbl5_1 = dbl5(graph,res4,512,"dbl5_1"); //// to output
    auto net1 = dbl(graph,dbl5_1,256,1,1,PD_MODE_SAME,"net");
    auto up1 = graph->upsampling(net1,2,2);
    auto concat1 = graph->concat({up1,res8_2},1);

    auto dbl5_2 = dbl5(graph,concat1,256,"dbl5_2");/// to output
    auto net2 = dbl(graph,dbl5_2,128,1,1,PD_MODE_SAME,"net2");
    auto up2 = graph->upsampling(net2,2,2);
    auto concat2 = graph->concat({up2,res8},1);

    auto dbl5_3 = dbl5(graph,concat2,128,"dbl5_3");/// to output

    auto dbl3 = dbl(graph,dbl5_3,256,3,1,PD_MODE_SAME,"dbl3");
    auto conv3 = conv2d(graph,dbl3,255,1,1,PD_MODE_SAME,"conv3");
    auto y3 = bias_add(graph,conv3,"y3");
    y3 = graph->reshape(y3,{-1,3,85,size/8,size/8});
    vector<TensorHandle> out3;
    graph->split(y3,{2,4},2,out3);
    out3[0] = graph->sigmoid(out3[0]);
    out3[2] = graph->sigmoid(out3[2]);
    y3 = graph->concat(out3,2);
    y3 = graph->reshape(y3,{-1,255,size/8,size/8});////y3

    auto dbl2 = dbl(graph,dbl5_2,512,3,1,PD_MODE_SAME,"dbl2");
    auto conv2 = conv2d(graph,dbl2,255,1,1,PD_MODE_SAME,"conv2");
    auto y2 = bias_add(graph,conv2,"y2");
    y2 = graph->reshape(y2,{-1,3,85,size/16,size/16});
    vector<TensorHandle> out2;
    graph->split(y2,{2,4},2,out2);
    out2[0] = graph->sigmoid(out2[0]);
    out2[2] = graph->sigmoid(out2[2]);
    y2 = graph->concat(out2,2);
    y2 = graph->reshape(y2,{-1,255,size/16,size/16});////y2

    auto dbl1 = dbl(graph,dbl5_1,1024,3,1,PD_MODE_SAME,"dbl1");
    auto conv1 = conv2d(graph,dbl1,255,1,1,PD_MODE_SAME,"conv1");
    auto y1 = bias_add(graph,conv1,"y1");
    y1 = graph->reshape(y1,{-1,3,85,size/32,size/32});
    vector<TensorHandle> out1;
    graph->split(y1,{2,4},2,out1);
    out1[0] = graph->sigmoid(out1[0]);
    out1[2] = graph->sigmoid(out1[2]);
    y1 = graph->concat(out1,2);
    y1 = graph->reshape(y1,{-1,255,size/32,size/32});////y1
    graph->function({y3,y2,y1});
    return graph;
}

