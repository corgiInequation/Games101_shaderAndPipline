# Games101 Experiment 3

## Experiment Content

- The main objective of this experiment is to read and understand the entire code framework, gaining a clear understanding of the graphics rendering pipeline in rasterization. After that, two types of fragment shaders are implemented.
- `phong_fragment_shader`  
  - This function is relatively simple. According to the Blinn-Phong shading model, it computes the ambient, diffuse, and specular components and then sums them to obtain the final color.
- `texture_fragment_shader`  
  - This function is built upon the `phong_fragment_shader`. In the Phong shader, the object's color at every point is the same, and only lighting causes variations in shading. However, in the texture shader, the color at each point on the object is determined by a texture, requiring an additional step to fetch the texture coordinates and retrieve the corresponding color based on them.

## Experiment Result

- phong_fragment_shader
- <img src="https://github.com/corgiInequation/Games101_shaderAndPipline/blob/main/output2.png" alt="image-20250511134235050" style="zoom:50%;" />
- texture_fragment_shader
- <img src="https://github.com/corgiInequation/Games101_shaderAndPipline/blob/main/output.png" alt="image-20250511134302915" style="zoom:50%;" />

