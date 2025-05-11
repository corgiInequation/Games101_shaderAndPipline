# Games101 Experiment3

## 实验内容

- 本次实验主要是阅读整个代码框架，了解光栅化的整个图形渲染管线。之后，实现2种着色函数。
- phong_fragment_shader
  - 该函数的实现比较简单，根据Bling Phong的着色公式，将环境光照，diffuse分量， view分量这三个分量算出来相加即可
- texture_fragment_shader
  - 该函数的实现基于上述的phong_fragment_shader。phong_fragment_shader中，物体每个位置的颜色都是一样的，只是由于光线的原因，会有不同的着色效果。而在texture_fragment_shader中，物体每个位置的颜色，都由纹理决定，需要额外一步去获取纹理坐标，之后根据坐标得到相应的颜色



## 实验效果

- phong_fragment_shader
- <img src="C:\Users\i love china\AppData\Roaming\Typora\typora-user-images\image-20250511134235050.png" alt="image-20250511134235050" style="zoom:50%;" />
- texture_fragment_shader
- <img src="C:\Users\i love china\AppData\Roaming\Typora\typora-user-images\image-20250511134302915.png" alt="image-20250511134302915" style="zoom:50%;" />

