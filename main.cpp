#include <string>
#include <iostream>
#include <vector>

#include "cl.hpp"

int main()
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::cout << "Platform=" << platforms.size() << "\n";


    if (platforms.size() != 1)
    {
        return EXIT_FAILURE;
    }

    auto platform = platforms.front();
    auto platform_info = platform.getInfo<CL_PLATFORM_NAME>();

    std::cout << "Platform info=" << platform_info << "\n";

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);

    std::cout << "devices=" << devices.size() << "\n";

    if (devices.size() != 1)
    {
        return EXIT_FAILURE;
    }

    auto device = devices.front();
    auto device_info = device.getInfo<CL_DEVICE_VENDOR>();

    std::cout << "device=" << device_info << "\n";

    std::string hello = R"(
__kernel void hello(__global char* data)
{
    data[0] = 'H';
    data[1] = 'E';
    data[2] = 'L';
    data[3] = 'L';
    data[4] = 'O';
}
)";

    std::string array = R"(
__kernel void array(__global int* data, __global int* output)
{
    output[get_global_id(0)] = data[get_global_id(0)] * 2;
}
)";

    cl::Program::Sources sources(1, std::make_pair(array.c_str(), array.size() + 1));
    cl::Context context(device);
    cl::Program program(context, sources);

    auto error = program.build(devices);

    std::cout << "build=" << error << "\n";

    std::vector<int> data{ 1, 2, 3, 4, 5 };
    cl::Buffer data_buffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_NO_ACCESS, sizeof(int) * data.size(), data.data(), &error);
    std::cout << "data_buffer=" << error << "\n";

    std::vector<int> output;
    output.resize(data.size());
    cl::Buffer output_buffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR | CL_MEM_HOST_READ_ONLY, sizeof(int) * output.size(), output.data(), &error);
    std::cout << "output_buffer=" << error << "\n";

    cl::Kernel kernel(program, "array", &error);
    std::cout << "kernel=" << error << "\n";

    error = kernel.setArg(0, data_buffer);
    std::cout << "kernel_setArg1=" << error << "\n";
    error = kernel.setArg(1, output_buffer);
    std::cout << "kernel_setArg2=" << error << "\n";

    cl::CommandQueue queue(context, device);
    error = queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(data.size()));
    std::cout << "enqueueNDRangeKernel=" << error << "\n";

    queue.enqueueMapBuffer(output_buffer, CL_TRUE, CL_MAP_READ, 0, sizeof(int) * output.size(), {}, {}, &error);
    std::cout << "enqueueMapBuffer=" << error << "\n";

    for (auto x : output)
    {
        std::cout << x << " ";
    }

    std::cout << "\n";

	return EXIT_SUCCESS;
}
