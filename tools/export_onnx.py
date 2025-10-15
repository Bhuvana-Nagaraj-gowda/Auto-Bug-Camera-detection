
import torch, argparse, os
from models.autobugnet import UNetSmall
from data.dataset import CLASSES

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=None)
    ap.add_argument("--out", default="runs/model.onnx")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()

    device = "cpu"
    net = UNetSmall(in_ch=3, seg_ch=len(CLASSES)).to(device)
    if args.weights and os.path.isfile(args.weights):
        net.load_state_dict(torch.load(args.weights, map_location=device))
    net.eval()

    dummy = torch.randn(1,3,args.size,args.size)
    torch.onnx.export(net, dummy, args.out, input_names=["image"], output_names=["seg","cls"], opset_version=17)
    print(f"Exported to {args.out}")
