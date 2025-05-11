#include <iostream>
#include <opencv2/opencv.hpp>
#include<cmath>
#include<algorithm>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1,0,0,-eye_pos[0],
                 0,1,0,-eye_pos[1],
                 0,0,1,-eye_pos[2],
                 0,0,0,1;
    view = translate*view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
                0, 1, 0, 0,
                -sin(angle), 0, cos(angle), 0,
                0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
              0, 2.5, 0, 0,
              0, 0, 2.5, 0,
              0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1;

    return translate * rotation * scale;
}

//投影变化
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
    // TODO: Copy-paste your implementation from the previous assignment.
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f P2O = Eigen::Matrix4f::Identity();
    //记得把zNear和zFar改过来，它这个框架比较难绷
    zNear *= -1;
    zFar *= -1;
    //将透视投影转换为正交投影的矩阵
    P2O<<zNear, 0, 0, 0,
    0, zNear, 0, 0,
    0, 0, zNear+zFar,(-1)*zFar*zNear,
    0, 0, 1, 0;
    
    float halfEyeAngelRadian = eye_fov/2.0/180.0*MY_PI;
    //记得给zNear加上负号（取绝对值），因为高和宽肯定是正数
    float t = -zNear*std::tan(halfEyeAngelRadian);//top y轴的最高点
    float r=t*aspect_ratio;//right x轴的最大值
    float l=(-1)*r;//left x轴最小值
    float b=(-1)*t;//bottom y轴的最大值
    
    //进行一定的缩放使之成为一个标准的长度为2的正方体
    Eigen::Matrix4f ortho1=Eigen::Matrix4f::Identity();
    ortho1<<2/(r-l),0,0,0,
    0,2/(t-b),0,0,
    0,0,2/(zNear-zFar),0,
    0,0,0,1;
    // 把一个长方体的中心移动到原点
    Eigen::Matrix4f ortho2 = Eigen::Matrix4f::Identity();
    ortho2<<1,0,0,(-1)*(r+l)/2,
    0,1,0,(-1)*(t+b)/2,
    0,0,1,(-1)*(zNear+zFar)/2,
    0,0,0,1;
    Eigen::Matrix4f Matrix_ortho = ortho1 * ortho2;
    projection = Matrix_ortho * P2O;

    return projection;
}

//返回的就是一个顶点的坐标？
Eigen::Vector3f vertex_shader(const vertex_shader_payload& payload)
{
    return payload.position;
}

//f_s_payload存储的是一个顶点的坐标，颜色，法案向量，以及材质，还有这个点在材质上的(u,v)
Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f& vec, const Eigen::Vector3f& axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        float u = payload.tex_coords.x();
        float v = payload.tex_coords.y();
        return_color = payload.texture->getColor(u,v);   
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    //result_color += ka.cwiseProduct(amb_light_intensity);

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        float distance = (point - light.position).norm();
        Eigen::Vector3f I_div_r2 =(light.intensity/std::pow(distance,2.0));
        Eigen::Vector3f h = (light.position - point + eye_pos - point)/ (light.position - point + eye_pos - point).norm();
        Eigen::Vector3f in = light.position - point;
        //这三个之所以需要化成单位向量，是因为，其之后会有点乘，表达的是一个角度，所以必须得规范化，不然会大于1
        normal.normalize();
        h.normalize();
        in.normalize();
        result_color += kd.cwiseProduct(I_div_r2)*std::max(0.0f, normal.dot(in)) + ks.cwiseProduct(I_div_r2)*std::pow(std::max(0.0f,normal.dot(h)),p);
    }

    return result_color * 255.f;
}


Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    //kd取决于材质，ks和ka取决于什么？
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    result_color += amb_light_intensity.cwiseProduct(ka);
    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        float distance = (point - light.position).norm();
        Eigen::Vector3f I_div_r2 =(light.intensity/std::pow(distance,2.0));
        Eigen::Vector3f h = (light.position - point + eye_pos - point)/ (light.position - point + eye_pos - point).norm();
        Eigen::Vector3f in = light.position - point;
        //这三个之所以需要化成单位向量，是因为，其之后会有点乘，表达的是一个角度，所以必须得规范化，不然会大于1
        normal.normalize();
        h.normalize();
        in.normalize();
        result_color += kd.cwiseProduct(I_div_r2)*std::max(0.0f, normal.dot(in)) + ks.cwiseProduct(I_div_r2)*std::pow(std::max(0.0f,normal.dot(h)),p);
    }

    return result_color * 255.f;
}


//这个是把三角形的顶点的位置都换了
Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload& payload)
{
    
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    
    // TODO: Implement displacement mapping here
    //伪代码已经给你了，具体原理讲光线追踪的时候会讲
    Eigen::Vector3f n = normal;
    float x = normal.x();
    float y = normal.y();
    float z = normal.z();  
    Eigen::Vector3f t; 
    t << x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z);
    Eigen::Vector3f b = n.cross(t);
    Eigen:: Matrix3f TBN;
    TBN.col(0) = t;
    TBN.col(1) = b;
    TBN.col(2) = n;
    //非常值得注意的一点，传入getColor中的数据都需要clamp一下，到[0,1]范围内
    auto u = payload.tex_coords.x();
    auto v = payload.tex_coords.y();
    u = std:: clamp(u, 0.0f, 1.0f);
    v = std:: clamp(v, 0.0f, 1.0f);
    auto w = payload.texture->width;
    auto h = payload.texture->height;
    auto temp1 = std::clamp(u + 1.0f / h, 0.0f, 1.0f);
    auto temp2 = std::clamp(v + 1.0f / h, 0.0f, 1.0f);

    auto dU = kh * kn * (payload.texture->getColor(u + 1.0f / w, v).norm() - payload.texture->getColor(u, v).norm());
    auto dV = kh * kn * (payload.texture->getColor(u, v + 1.0f / h).norm() - payload.texture->getColor(u, v).norm());

    Eigen::Vector3f ln{ -dU,-dV,1.0f };
    point += (kn * normal * payload.texture->getColor(u, v).norm()); //关键！将目标点拔高
    normal = TBN * ln;
    normal.normalized();
    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto& light : lights)
    {
        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular* 
        // components are. Then, accumulate that result on the *result_color* object.
        float distance = (point - light.position).norm();
        Eigen::Vector3f I_div_r2 =(light.intensity/std::pow(distance,2.0));
        Eigen::Vector3f h = (light.position - point + eye_pos - point)/ (light.position - point + eye_pos - point).norm();
        Eigen::Vector3f in = light.position - point;
        //这三个之所以需要化成单位向量，是因为，其之后会有点乘，表达的是一个角度，所以必须得规范化，不然会大于1
        normal.normalize();
        h.normalize();
        in.normalize();
        result_color += kd.cwiseProduct(I_div_r2)*std::max(0.0f, normal.dot(in)) + ks.cwiseProduct(I_div_r2)*std::pow(std::max(0.0f,normal.dot(h)),p);

    }

    return result_color * 255.f;
}


Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload& payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color; 
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;


    float kh = 0.2, kn = 0.1;

    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)
    auto x = normal.x();
    auto y = normal.y();
    auto z = normal.z();
    Eigen::Vector3f t(x * y / sqrt(x * x + z * z), sqrt(x * x + z * z), z * y / sqrt(x * x + z * z));
    Eigen::Vector3f b = normal.cross(t);
    Eigen::Matrix3f TBN; //TBN矩阵: 将纹理坐标对应到模型空间中
    TBN <<
        t.x(), b.x(), normal.x(),
        t.y(), b.y(), normal.y(),
        t.z(), b.z(), normal.z();
    
    auto u = payload.tex_coords.x();
    auto v = payload.tex_coords.y();
    u = std:: clamp(u, 0.0f, 1.0f);
    v = std:: clamp(v, 0.0f, 1.0f);
    auto w = payload.texture->width;
    auto h = payload.texture->height;
    auto temp1 = std::clamp(u + 1.0f / h, 0.0f, 1.0f);
    auto temp2 = std::clamp(v + 1.0f / h, 0.0f, 1.0f);

    auto dU = kh * kn * (payload.texture->getColor(temp1, v).norm() - payload.texture->getColor(u, v).norm());
    auto dV = kh * kn * (payload.texture->getColor(u, temp2).norm() - payload.texture->getColor(u, v).norm());
    

    Eigen::Vector3f ln{ -dU,-dV,1.0f };
    normal = TBN * ln;
    Eigen::Vector3f result_color = normal.normalized();

    return result_color * 255.f;
}

int main(int argc, const char** argv)
{
    std::vector<Triangle*> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "D:/learning_codes/games_work/duty_3/Code/models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("D:/learning_codes/games_work/duty_3/Code/models/spot/spot_triangulated_good.obj");
    if(loadout)
    {
        std::cout << "load success" << std::endl;
        std::cout << "LoadedMeshes size: " << Loader.LoadedMeshes.size() << std::endl;
    }
    for(auto mesh:Loader.LoadedMeshes)
    {
        for(int i=0;i<mesh.Vertices.size();i+=3)
        {
            Triangle* t = new Triangle();
            for(int j=0;j<3;j++)
            {
                //生成三角形的时候，三维空间中的三角形的三个点的坐标，法向量和对应的纹理坐标都会被加载
                //所以说无论用什么纹理，其u,v都是一样的？
                t->setVertex(j,Vector4f(mesh.Vertices[i+j].Position.X,mesh.Vertices[i+j].Position.Y,mesh.Vertices[i+j].Position.Z,1.0));
                t->setNormal(j,Vector3f(mesh.Vertices[i+j].Normal.X,mesh.Vertices[i+j].Normal.Y,mesh.Vertices[i+j].Normal.Z));
                t->setTexCoord(j,Vector2f(mesh.Vertices[i+j].TextureCoordinate.X, mesh.Vertices[i+j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }
    std::cout << "trianglelist size: " << TriangleList.size() << std::endl;

    rst::rasterizer r(700, 700);

    auto texture_path = "hmap.jpg";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = bump_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }
    Eigen::Vector3f eye_pos = {0,0,10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));
        r.draw(TriangleList);
        std::cout << "here" << std::endl;
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
        cv::imwrite(filename, image);

        return 0;
    }


    while(key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        //r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);

        if (key == 'a' )
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }

    }
    return 0;
}
