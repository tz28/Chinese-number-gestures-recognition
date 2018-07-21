package com.example.hc.digitalgesturerecognition;


import android.app.Activity;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;

public class DisplayResult extends Activity {
	//private MyDialog dialog;
	private LinearLayout layout;
	private ImageView imageView;
	private TextView reslutTextView;
	private TextView probTextView;
	private Bitmap bitmap;
	@Override
	protected void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.display_result);
		//bundle接受数据
		Bundle bundle = getIntent().getExtras();
		bitmap = (Bitmap)bundle.getParcelable("image");
		ArrayList<String> list = (ArrayList<String>)bundle.get("recognize_result");

		//展示图片
		imageView = (ImageView)findViewById(R.id.display_photo);
		imageView.setImageBitmap(bitmap);
		//展示识别结果
		reslutTextView = (TextView)findViewById(R.id.display_result);
		reslutTextView.setText("识别结果: " + list.get(0));
		probTextView = (TextView)findViewById(R.id.display_prob);
		probTextView.setText("识别概率: " + list.get(1));
		//dialog=new MyDialog(this);
		layout=(LinearLayout)findViewById(R.id.display_layout);
		layout.setOnClickListener(new OnClickListener() {
			
			@Override
			public void onClick(View v) {
				// TODO Auto-generated method stub
				Toast.makeText(getApplicationContext(), "hhhhhhh",
						Toast.LENGTH_SHORT).show();	
			}
		});
	}

	@Override
	public boolean onTouchEvent(MotionEvent event){
		finish();
		return true;
	}
	
	public void exitbutton1(View v) {  
    	this.finish();    	
      }
}
