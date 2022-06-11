//
// Created by sun on 2022/1/4.
//
#include"layer.h"
/*
 * is that right?
 */
static TensorHandle output38;
static TensorHandle output76;

TensorHandle cbl(Graph* graph, TensorHandle input, int oc, int kernel, int stride, const string& name=""){
    PaddingMode p;
    if(stride==1) p = PD_MODE_SAME;
    else p = PD_MODE_VALID;
    auto body = conv2d(graph,input,oc,kernel,stride,p,name+"_conv");
    body = batch_norm_infer(graph,body,1e-4,name+"_bn");
    return graph->leaky_relu(body);
}

TensorHandle res_unit(Graph* graph, TensorHandle input,int oc,const string& name=""){
    auto dbl1 = cbl(graph,input,oc,1,1,name+"_dbl1");
    auto dbl2 = cbl(graph,dbl1,oc,3,1,name+"_dl2");
    return graph->add(dbl2,input);
}

TensorHandle csp1(Graph* graph, TensorHandle input,int oc,int num_blocks,const string& name=""){
    auto left = cbl(graph,input,oc,1,1,name);
    for(int i=0;i<num_blocks;i++){
        left = res_unit(graph,left,oc,name+::to_string(i));
    }
    left = conv2d(graph,left,oc,1,1,PD_MODE_SAME,name);
    auto right = conv2d(graph,input,oc,1,1,PD_MODE_SAME,name);
    auto cat = graph->concat({left,right},1);
    auto bn = batch_norm_infer(graph,cat,1e-4,name);
    auto act = graph->leaky_relu(bn);
    return cbl(graph,act,oc,1,1,name);
}
TensorHandle csp2(Graph* graph, TensorHandle input,int oc,int num_blocks,const string& name=""){
    auto left = cbl(graph,input,oc,1,1,name);
    for(int i=0;i<num_blocks*2;i++){
        left = cbl(graph,left,oc,1,1,name+::to_string(i));
    }
    left = conv2d(graph,left,oc,1,1,PD_MODE_SAME,name);
    auto right = conv2d(graph,input,oc,1,1,PD_MODE_SAME,name);
    auto cat = graph->concat({left,right},1);
    auto bn = batch_norm_infer(graph,cat,1e-4,name);
    auto act = graph->leaky_relu(bn);
    return cbl(graph,act,oc,1,1,name);
}
TensorHandle focus(Graph* graph,TensorHandle input){
    int H = input->shape[2].get_upper_bound();
    int W = input->shape[3].get_upper_bound();
    auto slice1 = graph->slice(input,{0,0},{H,W},{2,3},{2,2});
    auto slice2 = graph->slice(input,{0,1},{H,W},{2,3},{2,2});
    auto slice3 = graph->slice(input,{1,0},{H,W},{2,3},{2,2});
    auto slice4 = graph->slice(input,{1,1},{H,W},{2,3},{2,2});
    auto cat = graph->concat({slice1,slice2,slice3,slice4},1);
    return cbl(graph,cat,48,3,1);
}
TensorHandle spp(Graph* graph, TensorHandle input,int oc,const string& name=""){
    auto y19 = cbl(graph,input,oc,1,1,"");
    auto pool1 = graph->max_pool2d(y19,13,13,1,1,PD_MODE_SAME);
    auto pool2 = graph->max_pool2d(y19,9,9,1,1,PD_MODE_SAME);
    auto pool3 = graph->max_pool2d(y19,5,5,1,1,PD_MODE_SAME);
    y19 = graph->concat({pool1,pool2,pool3,y19},1);
    y19 = cbl(graph,y19,oc,1,1,"");
    return y19;
}

TensorHandle darknet_body(Graph* graph,TensorHandle input){
    /*
     * backbone
     */
    auto x = focus(graph,input);
    x = cbl(graph,x,32,3,1,"");
    x = csp1(graph,x,512,1,"");
    x = cbl(graph,x,32,3,1,"");
    x = csp1(graph,x,256,3,"");
    output76 = x;
    x = cbl(graph,x,32,3,1,"");
    x = csp1(graph,x,256,3,"");
    output38 = x;
    x = cbl(graph,x,32,3,1,"");
    x = spp(graph,x,256,"");
    //output19
    return x;
}

Graph* yolo_v5(){
    auto graph = new Graph;
    auto input = graph->new_input({1,3,608,608});
    auto darknet = darknet_body(graph,input);
    ////19x19 head
    auto y19 = csp2(graph,darknet,256,1,"");
    auto y19_upsample = graph->upsampling(y19,2,2);
    y19_upsample = cbl(graph,y19_upsample,256,1,1,"");
    ////38x38 head
    auto y38 = graph->concat({output38,y19_upsample},1);
    y38 = csp2(graph,y38,256,1,"");
    auto y38_upsample = graph->upsampling(y38,2,2);
    y38_upsample = cbl(graph,y38_upsample,256,3,1,"");
    ////76x76 head
    auto y76 = graph->concat({output76,y38_upsample},1);
    y76 = csp2(graph,y76,256,1,"");
    ////76x76 output
    auto y76_output = conv2d(graph,y76,255,1,1,PD_MODE_SAME,"");
    //// 38x38 output
    auto y76_downsample = cbl(graph,y76,256,3,2,"");
    y38 = graph->concat({y76_downsample,y38},1);
    y38 = csp2(graph,y38,256,1,"");
    auto y38_output = conv2d(graph,y38,255,1,1,PD_MODE_SAME,"");
    ////19x19 output
    auto y38_downsample = cbl(graph,y38,256,1,1,"");
    y19 = graph->concat({y38_downsample,y19},1);
    y19 = csp2(graph,y19,256,1,"");
    auto y19_output = conv2d(graph,y19,255,1,1,PD_MODE_SAME,"");
    graph->function({y19_output,y38_output,y76_output});
    //todo fix the meta-parameter and do post_process
    return graph;
}