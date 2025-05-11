#include<stdio.h>
#include<unistd.h>

int main()
{
    int pid = fork();
    printf("%d\n", pid);
    return 0;
}

