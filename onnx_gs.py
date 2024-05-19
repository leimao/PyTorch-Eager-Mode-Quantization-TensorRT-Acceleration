import onnx
import onnx_graphsurgeon as gs

if __name__ == "__main__":

    # Load the ONNX model
    onnx_model_filepath = "saved_models/resnet_quantized_cifar10.onnx"
    modified_onnx_model_filepath = "saved_models/resnet_quantized_cifar10_modified.onnx"

    model = onnx.load(onnx_model_filepath)

    # Create a Graph Surgeon graph from the ONNX model
    graph = gs.import_onnx(model)

    # Remove all the cast nodes between QuantizeLinear and DequantizeLinear nodes
    for node in graph.nodes:
        if node.op == "Cast" and node.i(0).op == "QuantizeLinear" and node.o(
                0).op == "DequantizeLinear":
            dequantize_node = node.o(0)
            quantize_node = node.i(0)
            dequantize_node.inputs[0] = quantize_node.outputs[0]
            node.outputs.clear()
            node.inputs.clear()

    # Fix all the Conv biases.
    for node in graph.nodes:
        if node.op == "Conv":
            # Check if the Conv node has a bias term
            has_bias = False
            if len(node.inputs) == 3:
                has_bias = True
            is_quantized = False
            if node.i(0).op == "DequantizeLinear" and node.i(0).i(
                    0).op == "QuantizeLinear" and node.i(
                        1).op == "DequantizeLinear":
                is_quantized = True
            bias_needs_fix = False
            if node.i(2).op == "DequantizeLinear":
                bias_needs_fix = True
            if has_bias and is_quantized and bias_needs_fix:
                bias_int = node.i(2).inputs[0].inputs[0].attrs["value"].values
                bias_scale = node.i(
                    2).inputs[1].inputs[0].attrs["value"].values
                bias_float = bias_int * bias_scale
                node.inputs[2] = gs.Constant(name=f"{node.name}_bias",
                                             values=bias_float)

    graph.cleanup()

    # Save the modified graph back to an ONNX model
    model = gs.export_onnx(graph)
    onnx.save(model, modified_onnx_model_filepath)
