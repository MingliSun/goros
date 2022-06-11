//
// Created by sun on 2022/1/6.
//

#include"Polynomial.h"
#include<algorithm>
#include <utility>
#include<cassert>
#include<stack>
#include"Parser.h"
base operator+(const base&a,const base&b){
    if(a.index()==0&&b.index()==0){//both int
        return get<int>(a)+get<int>(b);
    }
    if(a.index()==1&&b.index()==1){//both float
        return get<float>(a)+get<float>(b);
    }
    float tmp_a,tmp_b;
    if(a.index()==0) tmp_a = get<int>(a);
    else tmp_a = get<float>(a);
    if(b.index()==0) tmp_b = get<int>(b);
    else tmp_b = get<float>(b);
    // return float
    return tmp_a+tmp_b;
}
base operator*(const base&a,const base&b){
    if(a.index()==0&&b.index()==0){//both int
        return get<int>(a)*get<int>(b);
    }
    if(a.index()==1&&b.index()==1){//both float
        return get<float>(a)*get<float>(b);
    }
    float tmp_a,tmp_b;
    if(a.index()==0) tmp_a = get<int>(a);
    else tmp_a = get<float>(a);
    if(b.index()==0) tmp_b = get<int>(b);
    else tmp_b = get<float>(b);
    // return float
    return tmp_a*tmp_b;
}
ostream& operator<<(ostream& os ,const base& a){
    if(a.index()==0){
        os<< get<int>(a);
    }else{
        os<< get<float>(a);
    }
    return os;
}
Poly1d::Poly1d(const vector<base>& coef){
    /*
     *  exponent from 0 1 ...... n
     *  different from numpy
     */
    for(size_t i=0;i<coef.size();i++){
        this->coef[i] = coef[i];
    }

}

Poly1d Poly1d::operator*(const Poly1d &that) const {
    Poly1d result;
    for(auto& p:this->coef){
        for(auto& q:that.coef){
            result.update(p.first+q.first,p.second*q.second);
        }
    }
    return result;
}

void Poly1d::update(int key, const base & value) {
    if(coef.find(key)==coef.end()) coef[key] = value;
    else coef[key] = coef[key]+ value;
}

float Poly1d::get_coef(int key) {
    if(coef.find(key)==coef.end()) return 0;
    base result =  coef[key];
    if(result.index()==0){
        return get<int>(result);
    }
    return get<float>(result);
}

float Polynomial::evaluate(const map<string, float> &eval_vap) {
    assert(std::all_of(variables.begin(),variables.end(),[&eval_vap](auto& item){return eval_vap.find(item)!=eval_vap.end();}));
    stack<float> s;
    for(auto &item:postfix){
        if(item.type==CONSTANT) {
            if(item.constant.index()==0) s.push(get<int>(item.constant));
            else s.push(get<float>(item.constant));
        }
        else if(item.type==VARIABLE) s.push(eval_vap.at(item.symbol));
        else if(item.type==SYMBOL){
            int index = Parser::keywords.at(item.symbol);
            if(index==5){//get one operand
                //floor ceil exp log tan cos sin sqrt  abs
                auto value = s.top();
                s.pop();
                if(item.symbol=="~") s.push(-value);
                else if(item.symbol=="floor") s.push(floor(value));
                else if(item.symbol=="ceil") s.push(ceil(value));
                else if(item.symbol=="exp") s.push(exp(value));
                else if(item.symbol=="log") s.push(log(value));
                else if(item.symbol=="tan") s.push(tan(value));
                else if(item.symbol=="cos") s.push(cos(value));
                else if(item.symbol=="sin") s.push(sin(value));
                else if(item.symbol=="sqrt") s.push(sqrt(value));
                else if(item.symbol=="abs") s.push(abs(value));
                else throw exception();
            }else{
                auto right = s.top();
                s.pop();
                auto left = s.top();
                s.pop();
                switch (item.symbol[0]) {
                    case '+':
                        s.push(left+right);
                        break;
                    case '-':
                        s.push(left-right);
                        break;
                    case '*':
                        s.push(left*right);
                        break;
                    case '/':
                        assert(right!=0);
                        s.push(left/right);
                        break;
                    case '%':
                        assert(right!=0);
                        s.push(Math::modulo(left,right));
                        break;
                    case '^':
                        s.push(pow(left,right));
                        break;
                    case 'm':
                        if(item.symbol=="max") s.push(max(left,right));
                        else s.push(min(left,right));
                        break;
                    default:
                        std::cerr<< "unexpected symbol (only supported 6 operators which are add sub mul div mod neg)"<<endl;
                        assert(false);
                }
            }
        }
    }
    auto ans  = s.top();
    return ans;
}

Intermedia::Intermedia(string  name, const basic_string<char>& index)
    :intermediaType(INTERMEDIA_VARIABLE),  dataType(DT_FLOAT), name(std::move(name)),var(index) {

//    vector<string> tokens;
//    string delimiters = ",";
//    string::size_type lastPos = index.find_last_not_of(delimiters,0);
//    string::size_type pos = index.find_first_not_of(delimiters,lastPos);
//    if(lastPos==string::npos){
//        intermediaType = INTERMEDIA_VARIABLE;
//        var = index;
//        return;
//    }
//    while(pos!=string::npos||string::npos!=lastPos){
//        tokens.push_back(index.substr(lastPos,pos-lastPos));
//        lastPos = index.find_first_not_of(delimiters,pos);
//        pos = index.find_first_not_of(delimiters,lastPos);
//    }
//    tensor_index = std::move(tokens);
}
Intermedia::Intermedia(string name,const vector<string>& v,const string& tensor_name)
    :intermediaType(INTERMEDIA_ELEMENT), dataType(DT_FLOAT), name(std::move(name)){
    tensor_index = std::move(make_pair(tensor_name,v));
}

Intermedia::Intermedia(string name, const Affine &a) :intermediaType(INTERMEDIA_AFFINE),dataType(DT_INT32),name(std::move(name)),affine(a){

}

base Intermedia::evaluate(const map<string, float> &m) {
    return 0;
}

string Intermedia::to_string() const {
    return std::__cxx11::string();
}

Intermedia::Intermedia(string name, Polynomial v, DataType t) :intermediaType(INTERMEDIA_POLYNOMIAL),dataType(t),name(std::move(name)),polynomial(std::move(v)){

}
Intermedia::Intermedia(string name,string func_name,vector<string> parameter)
    :intermediaType(INTERMEDIA_CALL),dataType(DT_FLOAT),name(std::move(name)) {
    call[std::move(func_name)] = std::move(parameter);
}

Function::Function(string name, vector<string> signature):
    name(std::move(name)),signature(std::move(signature)){

}
