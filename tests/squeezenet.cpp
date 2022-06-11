//
// Created by sun on 2022/1/2.
//
/*
 * Symbol of SqueezeNet

Reference:
Iandola, Forrest N., et al.
"Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size." (2016).
 */
#include"layer.h"

TensorHandle conv_block(Graph* graph,TensorHandle input,int oc,int kernel=1,int stride=1,const string& name=""){
    auto conv = conv2d(graph,input,oc,kernel,stride,PD_MODE_SAME,name+"_conv");
    auto net = bias_add(graph,conv,name+"_conv");
    net = graph->relu(net);
    return net;
}

TensorHandle  make_fire(Graph* graph,TensorHandle input,int squeeze_channels,int expand1x1_channels,int expand3x3_channels, const string& name=""){
    auto net = conv_block(graph,input,squeeze_channels,1,1,name+"_input");
    auto left = conv_block(graph,net,expand1x1_channels,1,1,name+"_left");
    auto right = conv_block(graph,net,expand3x3_channels,3,1,name+"_right");
    net = graph->concat({left,right},1);
    return net;
}

Graph* squeeze_net(int batch_size,
                   const vector<int>& image_shape,
                   int num_classes,
                   const string& version){
    assert(version=="1.0"||version=="1.1");
    vector<int> data_shape = image_shape;
    data_shape.insert(data_shape.begin(),batch_size);
    auto graph = new Graph;
    auto net = graph->new_input(data_shape,DT_FLOAT,"data");
    if(version=="1.0"){
        net = conv_block(graph,net,96,7,2,"conv1");
        net = graph->max_pool2d(net,3,3,2,2,PD_MODE_SAME);
        net = make_fire(graph,net,16,64,64,"fire1");
        net = make_fire(graph,net,16,64,64,"fire2");
        net = make_fire(graph,net,32,128,128,"fire3");
        net = graph->max_pool2d(net,3,3,2,2,PD_MODE_SAME);
        net = make_fire(graph,net, 32, 128, 128, "fire4");
        net = make_fire(graph,net, 48, 192, 192, "fire5");
        net = make_fire(graph,net, 48, 192, 192, "fire6");
        net = make_fire(graph,net, 64, 256, 256, "fire7");
        net = graph->max_pool2d(net,3,3,2,2,PD_MODE_SAME);
        net = make_fire(graph,net,64,256,256,"fire8");
    }else{
        net = conv_block(graph,net,64,3,2,"conv1");
        net = graph->max_pool2d(net,3,3,2,2,PD_MODE_SAME);
        net = make_fire(graph,net, 16, 64, 64, "fire1");
        net = make_fire(graph,net, 16, 64, 64, "fire2");
        net = graph->max_pool2d(net, 3,3, 2,2,PD_MODE_SAME);
        net = make_fire(graph,net, 32, 128, 128, "fire3");
        net = make_fire(graph,net, 32, 128, 128, "fire4");
        net = graph->max_pool2d(net, 3,3,2,2,PD_MODE_SAME);
        net = make_fire(graph,net, 48, 192, 192, "fire5");
        net = make_fire(graph,net, 48, 192, 192, "fire6");
        net = make_fire(graph,net, 64, 256, 256, "fire7");
        net = make_fire(graph,net, 64, 256, 256, "fire8");
    }
    net = graph->dropout(net,0.5);
    net = conv_block(graph,net,num_classes,1,1,"conv_final");
    net = graph->global_avg_pool2d(net);
    net = graph->batch_flatten(net);
    net = graph->softmax(net);
    graph->function({net});
    return graph;
}

void test_squeeze_net(int batch_size=1, int num_classes=1000,
                      const string& version="1.0",
                      const vector<int>& image_shape={3, 224, 224}){
    auto graph = squeeze_net(batch_size,image_shape,num_classes,version);
    cout<<graph->to_string()<<endl;
    cout<<"==========================="<<endl;
    graph->optimize();
    cout<<graph->to_string()<<endl;
    graph->codegen_dot(string(__FUNCTION__ ).append(".dot"));
    graph->codegen_te(string(__FUNCTION__ ).append(".py"));
}