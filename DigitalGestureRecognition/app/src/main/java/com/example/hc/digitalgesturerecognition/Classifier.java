package com.example.hc.digitalgesturerecognition;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.media.ThumbnailUtils;
import android.util.Log;

import org.tensorflow.contrib.android.TensorFlowInferenceInterface;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.lang.Math;

public class Classifier {

    //模型中输入变量的名称
    private static final String inputName = "input_x";
    //模型中输出变量的名称
    private static final String outputName = "predict";
    //概率变量的名称
    private static final String probabilityName = "probability";
    //cnn输出层的数据
    private static final String outlayerName = "outlayer";
    //图片维度
    private static final int IMAGE_SIZE = 64;

    TensorFlowInferenceInterface inferenceInterface;


    static {
        //加载libtensorflow_inference.so库文件
        System.loadLibrary("tensorflow_inference");
        Log.e("tensorflow","libtensorflow_inference.so库加载成功");
    }
    Classifier(AssetManager assetManager, String modePath) {
        //初始化TensorFlowInferenceInterface对象
        inferenceInterface = new TensorFlowInferenceInterface(assetManager,modePath);
        Log.e("tf","TensoFlow模型文件加载成功");
    }

    //convert bitmap to array
    private float[] getPixels(Bitmap bitmap) {


        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        float[] floatValues = new float[IMAGE_SIZE * IMAGE_SIZE * 3];

        if (bitmap.getWidth() != IMAGE_SIZE || bitmap.getHeight() != IMAGE_SIZE) {
            // rescale the bitmap if needed
            bitmap = ThumbnailUtils.extractThumbnail(bitmap, IMAGE_SIZE, IMAGE_SIZE);
        }

        bitmap.getPixels(intValues,0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        for (int i = 0; i < intValues.length; ++i) {
            final int val = intValues[i];
            floatValues[i * 3] = Color.red(val) / 255.0f;
            floatValues[i * 3 + 1] = Color.green(val) / 255.0f;
            floatValues[i * 3 + 2] = Color.blue(val) / 255.0f;
        }
        return floatValues;
    }

    //实现softmax
    public void softmax(double[] x) {
        double max = 0.0;
        double sum = 0.0;

        for(int i=0; i < x.length; i++)
        {
            if(max < x[i])
            {
                max = x[i];
            }

        }

        for(int i=0; i < x.length; i++) {
            x[i] = Math.exp(x[i] - max);
            sum += x[i];
        }

        for(int i=0; i<x.length; i++)
        {
            x[i] /= sum;
        }
    }


    public ArrayList predict(Bitmap bitmap)
    {
        ArrayList<String> list = new ArrayList<>();
        float[] inputdata = getPixels(bitmap);
        for(int i = 0; i <30; ++i)
        {
            Log.d("matrix",inputdata[i] + "");
        }
        inferenceInterface.feed(inputName, inputdata, 1, IMAGE_SIZE, IMAGE_SIZE, 3);
        //运行模型,run的参数必须是String[]类型
        String[] outputNames = new String[]{outputName,probabilityName,outlayerName};
        inferenceInterface.run(outputNames);
        //获取结果
        int[] labels = new int[1];
        inferenceInterface.fetch(outputName,labels);
        int label = labels[0];
        float[] prob = new float[11];
        inferenceInterface.fetch(probabilityName, prob);
//        float[] outlayer = new float[11];
//        inferenceInterface.fetch(outlayerName, outlayer);

//        for(int i = 0; i <11; ++i)
//        {
//            Log.d("matrix",outlayer[i] + "");
//        }
        for(int i = 0; i <11; ++i)
        {
            Log.d("matrix",prob[i] + "");
        }
        DecimalFormat df = new DecimalFormat("0.000000");
        float label_prob = prob[label];
        //返回值
        list.add(Integer.toString(label));
        list.add(df.format(label_prob));

        return list;
    }

}
