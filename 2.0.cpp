#include <iostream>
#include <string>
using namespace std;
//判断
bool is(char c)
{    //只要字母
    return(c>='A'&&c<='Z')||(c>='a'&&c<='z');
}
//递归双指针：只换字符
void rever(string &s,int l,int r)    
{
    if(l>=r)return;//终止条件
    //跳左边非字母
    if(!is(s[l])){//不是字母
        rever(s,l+1,r);//继续往右走
        return;
    }
    //跳右边非字母
    if(!is(s[r])){
        rever(s,l,r-1);
        return;
}
    //交换
    swap(s[l],s[r]);//交换变量 等价于下面三句
                    //char temp=s[l];
                   //s[l]=s[r];   
                  //s[r]=temp;
    //递归调用
    rever(s,l+1,r-1);
}
int main()
{
    string s;
    cout << "请输入字符串：" << flush;//flush刷新缓冲区
    if(!getline(cin,s))return 0;//从标准输入cin读取一行字符串，包含空格，遇到换行结束
    string s2=s;//备份原字符串
    if(!s.empty())//s.empty()是string的一个成员函数判断字符串是否为空
        rever(s,0,s.size()-1);//调用，给初值 s.size()返回字符串长度

    cout << "反转后字符串为：" << s << endl;
   
}