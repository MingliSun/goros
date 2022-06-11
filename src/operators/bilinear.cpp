//
// Created by sun on 2022/1/9.
//
#include"Ops.h"
#include"Polynomial.h"
vector<string> operator+(const vector<string>& a,const vector<string>& b){
    vector<string> ans =a;
    ans.insert(ans.end(),b.begin(),b.end());
    return ans;
}
Bilinear::Bilinear(TensorHandle _input,float scale_h,float scale_w, Graph* g)
    :OpBase(_input,OP_TOPK,g){
    vector<string> common = {_input->shape[0].name,_input->shape[1].name};
    vector<int> output_size={_input->shape[0].get_upper_bound(),_input->shape[1].get_upper_bound()};
    output_size.push_back(int(_input->shape[2].get_upper_bound()*scale_h));
    output_size.push_back(int(_input->shape[3].get_upper_bound()*scale_w));
    vector<Variable> output_shape = graph->get_variable(output_size);
    auto t = new Tensor;
    auto cpt = new Transform(t,_input,output_shape,graph->num_cpt,4);
    cpt->variable_table.insert(make_pair(Intermedia("in_x", Parser::make_polynomial({_input->shape[2].name, "/", scale_h})), 0));
    cpt->variable_table.insert(make_pair(Intermedia("in_y", Parser::make_polynomial({_input->shape[3].name, "/", scale_w})), 0));
    cpt->variable_table.insert(make_pair(Intermedia("left_x_index", Parser::make_polynomial("floor in_x"), DT_INT32), 0));
    cpt->variable_table.insert(make_pair(Intermedia("right_x_index", Parser::make_polynomial("ceil in_x"), DT_INT32), 0));
    cpt->variable_table.insert(make_pair(Intermedia("x_lerp", Parser::make_polynomial({"in_x", "-", "left_x_index"})), 0));

    cpt->variable_table.insert(make_pair(Intermedia("top_y_index", Parser::make_polynomial("floor in_y"), DT_INT32), 0));
    cpt->variable_table.insert(make_pair(Intermedia("bottom_y_index", Parser::make_polynomial("ceil in_y"), DT_INT32), 0));
    cpt->variable_table.insert(make_pair(Intermedia("y_lerp", Parser::make_polynomial({"in_y", "-", "top_y_index"})), 0));
    cpt->variable_table[Intermedia("top_left",common+ vector<string>({"top_y_index","left_x_index"}),_input->name)] = 0;

    cpt->variable_table[Intermedia("top_right",common+ vector<string>({"top_y_index","right_x_index"}),_input->name)] = 0;

    cpt->variable_table[Intermedia("bottom_left",common+ vector<string>({"bottom_y_index","left_x_index"}),_input->name)] = 0;

    cpt->variable_table[Intermedia("bottom_right",common+ vector<string>({"bottom_y_index","right_x_index"}),_input->name)] = 0;

    cpt->variable_table[Intermedia("top", Parser::make_polynomial("top_left*(1-y_lerp)+top_right*y_lerp"))] = 0;

    cpt->variable_table[Intermedia("bottom", Parser::make_polynomial("bottom_left*(1-y_lerp)+bottom_right*y_lerp"))] = 0;
    cpt->variable_table[Intermedia("value", Parser::make_polynomial("top*(1-x_lerp)+bottom*x_lerp"))] = 0;
    graph->push(cpt);
    outputs[0] = t;

}

Bicubic::Bicubic(TensorHandle _input,float scale_h,float scale_w, Graph* g):
    OpBase(_input,OP_WHERE,g){
    vector<int> output_size={_input->shape[0].get_upper_bound(),_input->shape[1].get_upper_bound()};
    output_size.push_back(int(_input->shape[2].get_upper_bound()*scale_h));
    output_size.push_back(int(_input->shape[3].get_upper_bound()*scale_w));
    vector<Variable> output_shape = graph->get_variable(output_size);
    auto t = new Tensor;
    auto cpt = new Transform(t,_input,output_shape,graph->num_cpt,16);
    Function cubic_kernel("cubic_kernel",{"A","B","C","t"});
    cubic_kernel.procedure.insert(Intermedia("a",Parser::make_polynomial("-A/2.0+3.0*B/2.0-3.0*C/2.0+D/2.0")));
    cubic_kernel.procedure.insert(Intermedia("b",Parser::make_polynomial("A-5.0*B/2.0+2.0*C-D/2.0")));
    cubic_kernel.procedure.insert(Intermedia("c",Parser::make_polynomial("-A/2.0+C/2.0")));
    cubic_kernel.procedure.insert(Intermedia("d","B"));
    cubic_kernel.procedure.insert(Intermedia("ans",Parser::make_polynomial("a * t * t * t + b * t * t + c * t + d")));
    cpt->function.insert(cubic_kernel);

    Function get_2d_pixel("get_2d_pixel",{"data", "image_height", "image_width", "n", "c", "h", "w"});
    get_2d_pixel.procedure.insert(Intermedia("y",Parser::make_polynomial("w min (image_width-1) max 0 ")));
    get_2d_pixel.procedure.insert(Intermedia("x",Parser::make_polynomial("h min (image_height-1) max 0 ")));
    get_2d_pixel.procedure.insert(Intermedia("ans",vector<string>({"n","c","x","y"}),"data"));
    cpt->function.insert(get_2d_pixel);

    cpt->variable_table[Intermedia("in_x", Parser::make_polynomial({_input->shape[2].name, "/", scale_h}))] =  0;
    cpt->variable_table[Intermedia("in_y", Parser::make_polynomial({_input->shape[3].name, "/", scale_w}))] =  0;
    cpt->variable_table[Intermedia("xint",Parser::make_polynomial("floor in_x"),DT_INT32)] = 0;
    cpt->variable_table[Intermedia("xfract",Parser::make_polynomial("in_x - xint"))] = 0;

    cpt->variable_table[Intermedia("yint",Parser::make_polynomial("floor in_y"),DT_INT32)] = 0;
    cpt->variable_table[Intermedia("yfract",Parser::make_polynomial("in_y - yint"))] = 0;
    vector<string> cc = {"data",_input->shape[2].name,_input->shape[3].name,_input->shape[0].name,_input->shape[1].name};
    //1st row
    cpt->variable_table[Intermedia("p00","get_2d_pixel",cc+vector<string>({"yint-1","xint-1"}))] = 0;
    cpt->variable_table[Intermedia("p10","get_2d_pixel",cc+vector<string>({"yint-1","xint+0"}))] = 0;
    cpt->variable_table[Intermedia("p20","get_2d_pixel",cc+vector<string>({"yint-1","xint+1"}))] = 0;
    cpt->variable_table[Intermedia("p30","get_2d_pixel",cc+vector<string>({"yint-1","xint+2"}))] = 0;
    //2st row
    cpt->variable_table[Intermedia("p01","get_2d_pixel",cc+vector<string>({"yint+0","xint-1"}))] = 0;
    cpt->variable_table[Intermedia("p11","get_2d_pixel",cc+vector<string>({"yint+0","xint+0"}))] = 0;
    cpt->variable_table[Intermedia("p21","get_2d_pixel",cc+vector<string>({"yint+0","xint+1"}))] = 0;
    cpt->variable_table[Intermedia("p31","get_2d_pixel",cc+vector<string>({"yint+0","xint+2"}))] = 0;
    //3st row
    cpt->variable_table[Intermedia("p02","get_2d_pixel",cc+vector<string>({"yint+1","xint-1"}))] = 0;
    cpt->variable_table[Intermedia("p12","get_2d_pixel",cc+vector<string>({"yint+1","xint+0"}))] = 0;
    cpt->variable_table[Intermedia("p22","get_2d_pixel",cc+vector<string>({"yint+1","xint+1"}))] = 0;
    cpt->variable_table[Intermedia("p32","get_2d_pixel",cc+vector<string>({"yint+1","xint+2"}))] = 0;
    //4st row
    cpt->variable_table[Intermedia("p03","get_2d_pixel",cc+vector<string>({"yint+2","xint-1"}))] = 0;
    cpt->variable_table[Intermedia("p13","get_2d_pixel",cc+vector<string>({"yint+2","xint+0"}))] = 0;
    cpt->variable_table[Intermedia("p23","get_2d_pixel",cc+vector<string>({"yint+2","xint+1"}))] = 0;
    cpt->variable_table[Intermedia("p33","get_2d_pixel",cc+vector<string>({"yint+2","xint+2"}))] = 0;

    cpt->variable_table[Intermedia("col0","cubic_kernel",vector<string>({"p00","p10","p20","p30","yfract"}))] =0;
    cpt->variable_table[Intermedia("col1","cubic_kernel",vector<string>({"p01","p11","p21","p31","yfract"}))] =0;
    cpt->variable_table[Intermedia("col2","cubic_kernel",vector<string>({"p02","p12","p22","p32","yfract"}))] =0;
    cpt->variable_table[Intermedia("col3","cubic_kernel",vector<string>({"p03","p13","p23","p33","yfract"}))] =0;

    cpt->variable_table[Intermedia("ans","cubic_kernel",vector<string>({"col0","col1","col2","col3","xfract"}))] = 0;
    graph->push(cpt);
    outputs[0] = t;
}

Trilinear::Trilinear(TensorHandle _input,float scale_d,float scale_h,float scale_w, Graph* g)
    :OpBase(_input,OP_WHERE,g){
    vector<int> output_size={_input->shape[0].get_upper_bound(),_input->shape[1].get_upper_bound()};
    output_size.push_back(int(_input->shape[2].get_upper_bound()*scale_d));
    output_size.push_back(int(_input->shape[3].get_upper_bound()*scale_h));
    output_size.push_back(int(_input->shape[4].get_upper_bound()*scale_w));
    vector<Variable> output_shape = graph->get_variable(output_size);
    auto t = new Tensor;
    auto cpt = new Transform(t,_input,output_shape,graph->num_cpt,8);

    Function get_pixel("get_pixel", {"data", "image_depth" , "image_height", "image_width", "n", "c", "d", "h", "w"});
    get_pixel.procedure.insert(Intermedia("z", Parser::make_polynomial("d min (image_depth-1) max 0 ")));
    get_pixel.procedure.insert(Intermedia("y", Parser::make_polynomial("w min (image_width-1) max 0 ")));
    get_pixel.procedure.insert(Intermedia("x", Parser::make_polynomial("h min (image_height-1) max 0 ")));
    get_pixel.procedure.insert(Intermedia("ans", vector<string>({"n", "c", "z","x", "y"}), "data"));
    cpt->function.insert(get_pixel);

    Function lerp("lerp",{"A","B","t"});
    lerp.procedure.insert(Intermedia("ans",Parser::make_polynomial("A*(1.0-t)+B*t")));
    cpt->function.insert(lerp);

    cpt->variable_table[Intermedia("in_x", Parser::make_polynomial({_input->shape[2].name, "/", scale_h}))] =  0;
    cpt->variable_table[Intermedia("in_y", Parser::make_polynomial({_input->shape[3].name, "/", scale_w}))] =  0;
    cpt->variable_table[Intermedia("in_z", Parser::make_polynomial({_input->shape[4].name, "/", scale_d}))] =  0;

    cpt->variable_table[Intermedia("xint",Parser::make_polynomial("floor in_x"),DT_INT32)] = 0;
    cpt->variable_table[Intermedia("xfract",Parser::make_polynomial("in_x - xint"))] = 0;

    cpt->variable_table[Intermedia("yint",Parser::make_polynomial("floor in_y"),DT_INT32)] = 0;
    cpt->variable_table[Intermedia("yfract",Parser::make_polynomial("in_y - yint"))] = 0;

    cpt->variable_table[Intermedia("zint",Parser::make_polynomial("floor in_z"),DT_INT32)] = 0;
    cpt->variable_table[Intermedia("zfract",Parser::make_polynomial("in_z - zint"))] = 0;

    vector<string> cc = {"data",_input->shape[2].name,_input->shape[3].name,_input->shape[4].name,_input->shape[0].name,_input->shape[1].name};

    cpt->variable_table[Intermedia("p000","get_pixel",cc+vector<string>({"zint","yint","xint"}))] = 0;
    cpt->variable_table[Intermedia("p001","get_pixel",cc+vector<string>({"zint","yint","xint+1"}))] = 0;
    cpt->variable_table[Intermedia("p010","get_pixel",cc+vector<string>({"zint","yint+1","xint"}))] = 0;
    cpt->variable_table[Intermedia("p011","get_pixel",cc+vector<string>({"zint","yint+1","xint+1"}))] = 0;
    cpt->variable_table[Intermedia("p100","get_pixel",cc+vector<string>({"zint+1","yint","xint"}))] = 0;
    cpt->variable_table[Intermedia("p101","get_pixel",cc+vector<string>({"zint+1","yint","xint+1"}))] = 0;
    cpt->variable_table[Intermedia("p110","get_pixel",cc+vector<string>({"zint+1","yint+1","xint"}))] = 0;
    cpt->variable_table[Intermedia("p111","get_pixel",cc+vector<string>({"zint+1","yint+1","xint+1"}))] = 0;

    cpt->variable_table[Intermedia("dep00","lerp",vector<string>({"p000","p100","zfract"}))] =0;
    cpt->variable_table[Intermedia("dep01","lerp",vector<string>({"p001","p101","zfract"}))] =0;
    cpt->variable_table[Intermedia("dep10","lerp",vector<string>({"p010","p110","zfract"}))] =0;
    cpt->variable_table[Intermedia("dep11","lerp",vector<string>({"p011","p111","zfract"}))] =0;

    cpt->variable_table[Intermedia("col0","lerp",vector<string>({"dep00","dep01","yfract"}))] =0;
    cpt->variable_table[Intermedia("col1","lerp",vector<string>({"dep10","dep11","yfract"}))] =0;
    cpt->variable_table[Intermedia("ans","lerp",vector<string>({"col0","col1","xfract"}))] =0;
    graph->push(cpt);
    outputs[0] = t;
}

