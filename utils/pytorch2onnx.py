from torch.autograd import Variable
import torch.onnx
import torchvision

# Export model to onnx format
model_name = "googlenet"
onnx_name = "{}.onnx".format(model_name)
model = None
export_torch = True
size = 224

if model_name == "resnet18":
    model = torchvision.models.resnet18(pretrained=True)
elif model_name == "resnet50":
    model = torchvision.models.resnet50(pretrained=True)
elif model_name == "squeezenet1_0":
    model = torchvision.models.squeezenet1_0(pretrained=True)
elif model_name == "mobilenet_v2":
    model = torchvision.models.mobilenet_v2(pretrained=True)
elif model_name == "inception_v3":
    size = 299
    model = torchvision.models.inception_v3(pretrained=True, progress=True,
                                            transform_input=False, aux_logits=False)
elif model_name == "googlenet":
    model = torchvision.models.googlenet(pretrained=True, progress=True,
                                         transform_input=False, aux_logits=False)
    # aux_logits cause trace error
    print(model)
else:
    print("model name not defined")
    exit(-1)


if model:
    dummy_input = Variable(torch.randn(1, 3, size, size))
    
    if export_torch:
        x = torch.randn(1, 3, size, size, requires_grad=True)
        model_ex = torch.jit.trace(model, x)
        model_ex.save("{}.pt".format(model_name))

    print(model)
    torch.onnx.export(model, dummy_input, onnx_name, opset_version=7)

    import onnx
    model = onnx.load(onnx_name)
    onnx.checker.check_model(model)
    print(onnx.helper.printable_graph(model.graph))