#include "keras_model.h"

int main() {
    // Initialize model.
    KerasModel model;
    model.LoadModel("example.model");
    // Create a 1D Tensor on length 10 for input data.
    Tensor in(1);
    in.data_ = {{10}};

    // Run prediction.
    Tensor out;
    model.Apply(&in, &out);
    out.Print();
    //system('pause');
    return 0;
}