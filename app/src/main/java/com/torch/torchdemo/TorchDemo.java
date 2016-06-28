package com.torch.torchdemo;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.view.KeyEvent;
import android.view.inputmethod.EditorInfo;
import android.view.inputmethod.InputMethodManager;
import android.widget.EditText;
import android.widget.TextView;
import android.util.Log;
import android.view.View;
import com.torch.Torch;
import android.content.Intent;

public class TorchDemo extends Activity
{
    /** Called when the activity is first created. */
    @Override
    public void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);
	    Log.d("torchdemo","Hello from JAVA\n");

        torch = Torch.getLuaManager(this);
	    //tv = new TextView(this);
        //tv.setText("Torch Created.");
        //setContentView(tv);
    }

    @Override
    public void onStart() {
        super.onStart();
        //torch.initTorch();
        Log.d("torchdemo", "onStart\n");
        Runnable r = new Runnable() {
            public void run() {
                //String returnFromC = torch.call("main.lua");
                //tv.setText(returnFromC);
                //                    setContentView(tv);
            }
        };
        r.run();
        editText = (EditText) findViewById(R.id.editText);
        textView = (TextView) findViewById(R.id.textView);
        editText.setOnEditorActionListener(new EditText.OnEditorActionListener() {
            @Override
            public boolean onEditorAction(TextView v, int actionId, KeyEvent event) {
                if (actionId == EditorInfo.IME_ACTION_DONE) {
                    translate();
                    InputMethodManager mgr = (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                    mgr.hideSoftInputFromWindow(editText.getWindowToken(), 0);
                    return true;
                }
                return false;
            }
        });
    }

    public void translate() {
        //Intent intent = new Intent(this, DisplayMessageActivity.class);
        textView = (TextView) findViewById(R.id.textView);
        editText = (EditText) findViewById(R.id.editText);
        Runnable r = new Runnable() {
            public void run() {

                String returnFromC = torch.translateSentence(editText.getText().toString());//torch.call("main.lua");
                textView.setText(returnFromC);
                //                    setContentView(tv);
            }
        };
        r.run();
    }

    //TextView tv;
    Torch torch;
    EditText editText;
    TextView textView;
}
