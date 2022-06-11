//
// Created by sun on 2021/12/17.
//

#include "Parser.h"
#include <sstream>
#include<iostream>
using namespace std;

const int error_code_this= -1;
list<Atom> Parser::infix;
list<Atom> Parser::postfix;
set<string> Parser::variables;
const map<string,int> Parser::keywords ={
make_pair("i",0),//constant
//affine keywords
make_pair("+",1),
make_pair("-",1),
make_pair("*",2),
make_pair("/",2),
make_pair("%",2),
make_pair("(",3),
make_pair(")",4),
make_pair("~",5),
make_pair("#",6),
//polynomial unary keywords
make_pair("floor",5),
make_pair("ceil",5),
make_pair("exp",5),
make_pair("log",5),
make_pair("tan",5),
make_pair("sin",5),
make_pair("cos",5),
make_pair("sqrt",5),
make_pair("abs",5),
make_pair("round",5),
//polynomial binary keywords
make_pair("max",7),
make_pair("min",7),
make_pair("^",7),//power

};


int Parser::get_index_in_op(const string& str){
    if(keywords.find(str)==keywords.end())    return error_code_this;
    return keywords.at(str);
}

int Parser::get_index_in_op(const Atom& item){
    assert(item.type!=ERROR);
    string str;
    if(item.type==CONSTANT|| item.type == VARIABLE) str = "i";
    else str = item.symbol;
    return get_index_in_op(str);
}


Compare Parser::compare_items(const string& ch1,const string& ch2){
    int ia = get_index_in_op(ch1),ib = get_index_in_op(ch2);
    assert(ia != error_code_this && ib != error_code_this);
    return compare[ia][ib];
}


void Parser::lex(const string& expr,map<string,string>& var_map,map<string,int>& eval){
    int n=expr.length();
    int i=0;
    string digit;
    string var;
    string sym;
    bool isfloat = false;
    while(i<n){
        char ch = expr[i];
        if(isalpha(ch)){
            var+=ch;
            while(++i<n&&(isalnum(ch = expr[i]))){
                var+=ch;
            }
            if(eval.find(var)!=eval.end()){//found
                Atom item(CONSTANT,eval[var]);
                infix.push_back(item);
            }else if(keywords.find(var)!=keywords.end()){
                Atom item(SYMBOL,var);
                infix.push_back(item);
            }else{
                if(var_map.find(var)!=var_map.end())  var = var_map[var];
                Atom item(VARIABLE,var);
                variables.insert(var);//keep record of variables
                infix.push_back(item);
            }
            var="";
        }
        else if(isdigit(ch,isfloat)){
            digit+=ch;
            while(++i<n&&isdigit(ch = expr[i],isfloat)){
                digit+=ch;
            }
            stringstream ss;
            ss << digit;
            if(isfloat){
                float f;
                ss>> f;
                Atom item(CONSTANT,f);
                infix.push_back(item);
            }else{
                int value;
                ss >> value;
                Atom item(CONSTANT,value);
                infix.push_back(item);
            }
            isfloat = false;
            digit="";
        }else if(keywords.find(string(1,ch))!=keywords.end()){ // single character keywords
            sym +=ch;
            Atom item(SYMBOL,sym);
            infix.push_back(item);
            sym="";
            i++;
        }else if(ch==' '){
            i++;
        }
        else{
            Atom item(ERROR);
            infix.push_back(item);
            i++;
        }
    }
    Atom item(END,"#");
    infix.push_back(item);
}

void Parser::infix2postfix(){
    stack<string> s;
    s.push("#");
    Atom old(END,"#"),now;
    while(!s.empty()){
        now = infix.front();
        infix.pop_front();
        //fix negative representation
        if(now.type==SYMBOL&&now.symbol=="-"&&!is_compatible(old,now)) now.symbol="~";
        assert(is_compatible(old,now));
        assert(now.type!=ERROR);
        if(now.type==CONSTANT||now.type==VARIABLE) postfix.push_back(now);
        while(now.type==SYMBOL||now.type==END){
            string s1 = s.top();
            string s2 = now.symbol;
            Compare com = compare_items(s1,s2);
            assert(com != COMPARE_ER);
            if(com==COMPARE_GT) {
                s.pop();
                Atom item(SYMBOL,s1);
                postfix.push_back(item);
            }else if(com==COMPARE_LT){
                s.push(now.symbol);
                break;
            }else if(com==COMPARE_EQ){
                s.pop();
                break;
            }
        }
        old = now;
    }
}

Affine Parser::make_affine(const string& expr,map<string,int>& eval) {
    map<string,string> var_map;
    return make_affine(expr,var_map,eval);
}

Affine Parser::make_affine() {
    stack<Affine> s;
    for(auto &item:postfix){
        if(item.type==CONSTANT) {
            assert(item.constant.index()==0);
            s.push(Affine(get<int>(item.constant)));
        }
        else if(item.type==VARIABLE) s.push(Affine(Variable(item.symbol)));
        else if(item.type==SYMBOL){
            if(item.symbol=="~"){//get one operand
                auto value = s.top();
                s.pop();
                s.push(value.neg());
            }else{
                auto right = s.top();
                s.pop();
                auto left = s.top();
                s.pop();
                switch (item.symbol[0]) {
                    case '+':
                        s.push(left.add(right));
                        break;
                    case '-':
                        s.push(left.sub(right));
                        break;
                    case '*':
                        s.push(left.mul(right));
                        break;
                    case '/':
                        assert(!right.is_zero());
                        s.push(left.div(right));
                        break;
                    case '%':
                        assert(!right.is_zero());
                        s.push(left.mod(right));
                        break;
                    default:
                        std::cerr<< "unexpected symbol (only supported 6 operators which are add sub mul div mod neg)"<<endl;
                        assert(false);
                }
            }
        }
    }
    auto ans  = s.top();
    ans.variables = variables;
    return ans;
}

//void Parser::print(bool flag){
//    if(flag){
//        for(auto & i : infix){
//            cout<< atom_type_string[i.type]<<" "<<i.constant<<" "<<i.symbol<<endl;
//        }
//    }else{
//        for(auto & i : postfix){
//            cout<< atom_type_string[i.type]<<" "<<i.constant<<" "<<i.symbol<<endl;
//        }
//    }
//}

Affine Parser::make_affine(const string &expr) {
    map<string,int> eval;
    return make_affine(expr,eval);
}

Affine Parser::make_affine(const vector<atomi> &v) {
    for(auto&i:v){
        if(i.index()==0){//string
            const string& temp = get<0>(i);
            if(keywords.find(temp)!=keywords.end()) infix.emplace_back(SYMBOL,temp);
            else {
                infix.emplace_back(VARIABLE, temp);
                variables.insert(temp);
            }
        }else{//int
            infix.emplace_back(CONSTANT,get<1>(i));
        }
    }
    infix.emplace_back(END,"#");
    infix2postfix();
    auto ans =  make_affine();
    infix.clear();
    postfix.clear();
    variables.clear();
    return ans;
}

void Parser::lex(const string& expr, map<string, int> &eval) {
    map<string,string> var_map;
    lex(expr,var_map,eval);
}

Affine Parser::make_affine(const string &expr, map<string, string> &var_map, map<string, int> &eval) {
    lex(expr,var_map,eval);
    infix2postfix();
    auto ans =  make_affine();
    infix.clear();
    postfix.clear();
    variables.clear();
    return ans;
}

bool Parser::is_compatible(Atom &old, Atom &now) {
    int ia = get_index_in_op(old);
    int ib = get_index_in_op(now);
    assert(ia != error_code_this);
    assert(ib != error_code_this);
    return compatibility[ia][ib];
}

Polynomial Parser::make_polynomial(const vector<atomf> &v) {
    for(auto&i:v){
        if(i.index()==0){//string
            const string& temp = get<0>(i);
            if(keywords.find(temp)!=keywords.end()) infix.emplace_back(SYMBOL,temp);
            else {
                infix.emplace_back(VARIABLE, temp);
                variables.insert(temp);
            }
        }else{//float
            infix.emplace_back(CONSTANT,get<1>(i));
        }
    }
    infix.emplace_back(END,"#");
    infix2postfix();
    Polynomial ans;
    ans.infix = infix;
    ans.postfix = postfix;
    ans.variables = variables;

    infix.clear();
    postfix.clear();
    variables.clear();
    return ans;
}

bool Parser::isalpha(char ch) {
    if(::isalpha(ch)||ch=='_'||ch=='$'||ch=='@') return true;
    return false;
}

bool Parser::isalnum(char ch) {
    if(::isalnum(ch)||ch=='_'||ch=='$'||ch=='@') return true;
    return false;
}

bool Parser::isdigit(char ch, bool& isfloat) {
    bool ans = false;
    if(::isdigit(ch)){
        isfloat |= false;
        ans = true;
    }else if(ch=='.'){
        isfloat = true;
        ans = true;
    }
    return ans;
}

Polynomial Parser::make_polynomial(const string& expr) {
    map<string,int> eval;
    map<string,string> var_map;
    lex(expr,var_map,eval);
    infix2postfix();

    Polynomial ans;
    ans.infix = infix;
    ans.postfix = postfix;
    ans.variables = variables;

    infix.clear();
    postfix.clear();
    variables.clear();
    return ans;
}
