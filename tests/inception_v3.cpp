//
// Created by sun on 2021/12/31.
//

#include"layer.h"

TensorHandle conv_block(Graph* graph,TensorHandle input,int oc,const string& name="",int kernel=1,int stride=1){
    auto conv = conv2d(graph,input,oc,kernel,stride,PD_MODE_SAME,name+"_conv1");
    auto bn = batch_norm_infer(graph,conv,2e-5,name+"_bn");
    auto act = graph->relu(bn);
    return act;
}
TensorHandle conv_block(Graph* graph,TensorHandle input,int oc,const vector<int>& kernel={1,1},int stride=1,const string& name=""){
    int ic = input->shape[1].get_upper_bound();
    TensorHandle weight = graph->new_weight({oc,ic,kernel[0],kernel[1]},DT_FLOAT,name+"_conv1_weight");
    auto conv =  graph->conv2d(input,weight,stride,stride,PD_MODE_SAME);
    auto bn = batch_norm_infer(graph,conv,2e-5,name+"_bn");
    auto act = graph->relu(bn);
    return act;
}

TensorHandle pooling(Graph* graph,TensorHandle input,int kernel,int stride,const string& pool_type){
    if(pool_type=="max")
        return graph->max_pool2d(input,kernel,kernel,stride,stride,PD_MODE_SAME);
    if(pool_type=="avg")
        return graph->avg_pool2d(input,kernel,kernel,stride,stride,PD_MODE_SAME);
    throw exception();
}

TensorHandle Inception7A(Graph* graph,TensorHandle input,
                         int num_1x1,
                         int num_3x3_red,
                         int num_3x3_1,
                         int num_3x3_2,
                         int num_5x5_red,
                         int num_5x5,
                         const string& pool,
                         int proj,
                         const string& name){
    auto tower_1x1 = conv_block(graph,input,num_1x1,name+"_conv");
    auto tower_5x5 = conv_block(graph,input,num_5x5_red,name+"_tower_conv");
    tower_5x5 = conv_block(graph,tower_5x5,num_5x5,name+"_tower_conv_1");
    auto tower_3x3 = conv_block(graph,input,num_3x3_red,name+"_tower_1_conv");
    tower_3x3 = conv_block(graph,tower_3x3,num_3x3_1,name+"_tower_1_conv_1",3);
    tower_3x3 = conv_block(graph,tower_3x3,num_3x3_2,name+"tower_1_conv_2",3);
    auto p = pooling(graph,input,3,1,pool);
    auto cproj = conv_block(graph,p,proj,name+"_tower_2_conv");
    auto cat = graph->concat({tower_1x1,tower_5x5,tower_3x3,cproj},1);
    return cat;
}
//first Down sample
TensorHandle Inception7B(Graph* graph,TensorHandle input,int num_3x3,int num_d3x3_red,int num_d3x3_1,int num_d3x3_2,const string& pool,const string& name){
    auto tower_3x3 = conv_block(graph,input,num_3x3,name+"_conv",3,2);
    auto tower_d3x3 = conv_block(graph,input,num_d3x3_red,name+"_tower_conv");
    tower_d3x3 = conv_block(graph,tower_d3x3,num_d3x3_1,name+"_tower_conv_1",3,1);
    tower_d3x3 = conv_block(graph,tower_d3x3,num_d3x3_2,name+"_tower_conv_2",3,2);
    auto p = pooling(graph,input,3,2,"max");
    auto cat = graph->concat({tower_3x3,tower_d3x3,p},1);
    return cat;
}

TensorHandle Inception7C(Graph* graph,TensorHandle input,
                         int num_1x1,
                         int num_d7_red,
                         int num_d7_1,
                         int num_d7_2,
                         int num_q7_red,
                         int num_q7_1,
                         int num_q7_2,
                         int num_q7_3,
                         int num_q7_4,
                         const string& pool,
                         int proj,
                         const string& name){
    auto tower_1x1 = conv_block(graph,input,num_1x1,name+"_conv",1);
    auto tower_d7 = conv_block(graph,input,num_d7_red,name+"_tower_conv");
    tower_d7 = conv_block(graph,tower_d7,num_d7_1,{1,7},1,name+"_tower_conv_1");
    tower_d7 = conv_block(graph,tower_d7,num_d7_2,{7,1},1,name+"_tower_conv_2");
    auto tower_q7 = conv_block(graph,input,num_q7_red,name+"_tower_1_conv");
    tower_q7 = conv_block(graph,tower_q7,num_q7_1,{7,1},1,name+"_tower_1_conv_1");
    tower_q7 = conv_block(graph,tower_q7,num_q7_2,{1,7},1,name+"_tower_1_conv_2");
    tower_q7 = conv_block(graph,tower_q7,num_q7_3,{7,1},1,name+"_tower_1_conv_3");
    tower_q7 = conv_block(graph,tower_q7,num_q7_4,{1,7},1,name+"_tower_1_conv_4");
    auto p = pooling(graph,input,3,1,pool);
    auto cproj = conv_block(graph,p,proj,name+"_tower_2_conv");
    auto cat = graph->concat({tower_1x1,tower_d7,tower_q7,cproj},1);
    return cat;
}

TensorHandle Inception7D(Graph* graph,TensorHandle input,
                         int num_3x3_red,
                         int num_3x3,
                         int num_d7_3x3_red,
                         int num_d7_1,
                         int num_d7_2,
                         int num_d7_3x3,
                         const string& pool,
                         const string& name){
    auto tower_3x3 = conv_block(graph,input,num_3x3_red,name+"_tower_conv");
    tower_3x3 = conv_block(graph,tower_3x3,num_3x3,name+"_tower_conv_1",3,2);
    auto tower_d7_3x3 = conv_block(graph,input,num_d7_3x3_red,name+"_tower_1_conv");
    tower_d7_3x3 = conv_block(graph,tower_d7_3x3,num_d7_1,{1,7},1,name+"_tower_1_conv_1");
    tower_d7_3x3 = conv_block(graph,tower_d7_3x3,num_d7_2,{7,1},1,name+"_tower_1_conv_2");
    tower_d7_3x3 = conv_block(graph,tower_d7_3x3,num_d7_3x3,name+"_tower_1_conv_3",3,2);
    auto p = pooling(graph,input,3,2,pool);
    auto cat = graph->concat({tower_3x3,tower_d7_3x3,p},1);
    return cat;
}

TensorHandle Inception7E(Graph* graph,TensorHandle input,
                         int num_1x1,
                         int num_d3_red,
                         int num_d3_1,
                         int num_d3_2,
                         int num_3x3_d3_red,
                         int num_3x3,
                         int num_3x3_d3_1,
                         int num_3x3_d3_2,
                         const string& pool,
                         int proj,
                         const string& name){
    auto tower_1x1 = conv_block(graph,input,num_1x1,name+"_conv");
    auto tower_d3 = conv_block(graph,input,num_d3_red,name+"_tower_conv");
    auto tower_d3_a = conv_block(graph,tower_d3,num_d3_1,{1,3},1,name+"_tower_mixed_conv");
    auto tower_d3_b = conv_block(graph,tower_d3,num_d3_2,{3,1},1,name+"_tower_mixed_conv_1");
    auto tower_3x3_d3 = conv_block(graph,input,num_3x3_d3_red,name+"_tower_1_conv");
    tower_3x3_d3 = conv_block(graph,tower_3x3_d3,num_3x3,name+"_tower_1_conv_1",3,1);
    auto tower_3x3_d3_a = conv_block(graph,tower_3x3_d3,num_3x3_d3_1,{1,3},1,name+"_tower_1_mixed_conv");
    auto tower_3x3_d3_b = conv_block(graph,tower_3x3_d3,num_3x3_d3_2,{3,1},1,name+"_tower_1_mixed_conv_1");
    auto p = pooling(graph,input,3,1,pool);
    auto cproj = conv_block(graph,p,proj,name+"_tower_2_conv");
    auto cat = graph->concat({tower_1x1,tower_d3_a,tower_d3_b,tower_3x3_d3_a,tower_3x3_d3_b,cproj},1);
    return cat;

}

Graph* Inception_v3(int batch_size,int num_classes,const vector<int>& image_shape){
    vector<int> data_shape = image_shape;
    data_shape.insert(data_shape.begin(),batch_size);
    auto graph = new Graph;
    auto data = graph->new_input(data_shape,DT_FLOAT,"data");
    ////stage 1
    auto conv = conv_block(graph,data,32,"conv",3,2);
    auto conv_1 = conv_block(graph,conv,32,"conv_1",3);
    auto conv_2 = conv_block(graph,conv_1,64,"conv_2",3);
    auto p = pooling(graph,conv_2,3,2,"max");
    ////stage 2
    auto conv_3 = conv_block(graph,p,80,"conv_3");
    auto conv_4 = conv_block(graph,conv_3,192,"conv_4",3);
    auto p1 = pooling(graph,conv_4,3,2,"max");
    ////stage 3
    auto in3a = Inception7A(graph, p1, 64, 64, 96, 96, 48, 64, "avg", 32, "mixed");
    auto in3b = Inception7A(graph,in3a,64, 64, 96, 96, 48, 64, "avg", 64, "mixed_1");
    auto in3c = Inception7A(graph,in3b, 64, 64, 96, 96, 48, 64, "avg", 64, "mixed_2");
    auto in3d = Inception7B(graph,in3c, 384, 64, 96, 96, "max", "mixed_3");
    ////stage 4
    auto in4a = Inception7C(graph,in3d, 192, 128, 128, 192, 128, 128, 128, 128, 192, "avg", 192, "mixed_4");
    auto in4b = Inception7C(graph ,in4a, 192, 160, 160, 192, 160, 160, 160, 160, 192, "avg", 192, "mixed_5");
    auto in4c = Inception7C(graph,in4b, 192, 160, 160, 192, 160, 160, 160, 160, 192, "avg", 192, "mixed_6");
    auto in4d = Inception7C(graph,in4c, 192, 192, 192, 192, 192, 192, 192, 192, 192, "avg", 192, "mixed_7");
    auto in4e = Inception7D(graph,in4d, 192, 320, 192, 192, 192, 192, "max", "mixed_8");
    ////stage 5
    auto in5a = Inception7E(graph,in4e, 320, 384, 384, 384, 448, 384, 384, 384, "avg", 192, "mixed_9");
    auto in5b = Inception7E(graph,in5a, 320, 384, 384, 384, 448, 384, 384, 384, "max", 192, "mixed_10");
    ////pool
    auto pool = pooling(graph,in5b,8,1,"avg");
    auto flatten = graph->batch_flatten(pool);
    auto fc1 = dense_add_bias(graph,flatten,num_classes,"fc1");
    auto net = graph->softmax(fc1);
    graph->function({net});
    return graph;
}