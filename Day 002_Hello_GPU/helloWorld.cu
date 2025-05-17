__global__ void helloWorld() {
    printf("Hello World from GPU!\n");
}

int main() {
    helloWorld<<<1, 1>>>();
    return 0;
}
