import sys
import os
import cv2
import numpy as np
import tritonclient.grpc as grpcclient

if sys.version_info.major == 3:
    unicode = bytes


def simple_inference(triton_client):
    model_name = "image_process"
    inputs = []
    outputs = []
    inputs.append(grpcclient.InferInput("image_path", [2, 1], "BYTES"))
    in0n = np.array(
        [[os.path.join(os.getcwd(), "test_imgs/600001.jpg")], 
         [os.path.join(os.getcwd(),"test_imgs/331_1658719903000_01_960_1658719935779.jpg")]],
        dtype=np.object_
    )
    inputs[0].set_data_from_numpy(in0n)
    outputs.append(grpcclient.InferRequestedOutput("image"))
    results = triton_client.infer(model_name=model_name, inputs=inputs, outputs=outputs)

    print(results.as_numpy("image").shape)
    images = results.as_numpy("image")
    images = np.array(images, dtype=np.float32)
    print(images.max(), images.min())
    im = images[0, :, :, :]
    im1 = images[1, :, :, :]
    cv2.imwrite('re1.jpg', im)
    cv2.imwrite('re2.jpg', im1)



if __name__ == "__main__":


    triton_client = grpcclient.InferenceServerClient(
            url="localhost:8001", verbose=False
    )

    simple_inference(triton_client)