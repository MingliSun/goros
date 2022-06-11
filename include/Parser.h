//
// Created by sun on 2021/12/17.
//

#ifndef AFFINE_PARSER_H
#define AFFINE_PARSER_H
#include<cstring>
#include<string>
#include <utility>
#include<vector>
#include<stack>
#include<list>
#include<cassert>
#include<iostream>
#include<algorithm>
#include<map>
#include<set>
#include <variant>
#include"Affine.h"
#include"Producer.h"
#include"Polynomial.h"
using namespace std;
enum Compare{
    COMPARE_EQ,
    COMPARE_LT,
    COMPARE_GT,
    COMPARE_ER,

};
// ~ (negation)
//todo +-  */%  share same column and row -done

const bool compatibility[8][8]={
//    i     +     *    (     )     ~    #     ^
        
    false,true, true,false,true,false,true, true,
    true,false,false,true,false,true,false,false,
    true,false,false,true,false,true,false,false,
    true,false,false,true,false,true,false,false,
    false,true,true,false,true,false, true, true,
    true,false,false,true,false,true,false,false,
    true,false,false,true,false,true,false,false,
    true,false,false,true,false,true,false,false,
};
// sub and neg are opposite at compatibility matrix,so we can build - at lex phase and substitute - with ~ if - fails at compatibility
const Compare compare[8][8] = {
//                  i           +           *           (           )           ~           #          ^
/* i */        COMPARE_ER,COMPARE_ER,COMPARE_ER, COMPARE_ER, COMPARE_ER, COMPARE_ER, COMPARE_ER,COMPARE_ER,
/* + */        COMPARE_ER,COMPARE_GT,COMPARE_LT, COMPARE_LT, COMPARE_GT, COMPARE_LT, COMPARE_GT,COMPARE_LT,
/* * */        COMPARE_ER,COMPARE_GT,COMPARE_GT, COMPARE_LT, COMPARE_GT, COMPARE_LT, COMPARE_GT,COMPARE_LT,
/* ( */        COMPARE_ER,COMPARE_LT,COMPARE_LT, COMPARE_LT, COMPARE_EQ, COMPARE_LT, COMPARE_ER,COMPARE_LT,
/* ) */        COMPARE_ER,COMPARE_ER,COMPARE_ER, COMPARE_ER, COMPARE_ER, COMPARE_ER, COMPARE_ER,COMPARE_ER,
/* ~ */        COMPARE_ER,COMPARE_GT,COMPARE_GT, COMPARE_LT, COMPARE_GT, COMPARE_GT, COMPARE_GT,COMPARE_GT,
/* # */        COMPARE_ER,COMPARE_LT,COMPARE_LT, COMPARE_LT, COMPARE_ER, COMPARE_LT, COMPARE_EQ,COMPARE_LT,
/* ^ */        COMPARE_ER,COMPARE_GT,COMPARE_GT, COMPARE_LT, COMPARE_GT, COMPARE_LT, COMPARE_GT,COMPARE_GT,
};

using atomi = std::variant<string,int>;
using atomf = std::variant<string,float>;
class Parser {
public:
    static Affine make_affine(const string& expr,map<string,string>& var_map,map<string,int>& eval);
    static Affine make_affine(const string& expr,map<string,int>& eval);
    static Affine make_affine(const string& expr);
    static Affine make_affine(const vector<atomi>& v);
    static Polynomial make_polynomial(const vector<atomf>& v);
    static Polynomial make_polynomial(const string& expr);
    const static map<string,int> keywords;
private:
    /*
     *   floor ceil exp log tan cos sin sqrt  abs   : usages: floor x (same as neg)
     *   max min power : usages x max y(same as * ), it is different from max(x,y)
     */

    static list<Atom> infix;
    static list<Atom> postfix;
    static void lex(const string& expr,map<string,int>& eval);
    static void lex(const string& expr,map<string,string>& var_map,map<string,int>& eval);
    static void infix2postfix();
    static Affine make_affine();
//    static void print(bool infix=true);
    static bool is_compatible(Atom &old,Atom &now);
    static int get_index_in_op(const string& ch);
    static int get_index_in_op(const Atom& item);
    static Compare compare_items(const string& ch1,const string& ch2);
    //keep record of variable
    static set<string> variables;
    static bool isalpha(char ch);
    static bool isalnum(char ch);
    static bool isdigit(char ch, bool & isfloat );
};


#endif //AFFINE_PARSER_H