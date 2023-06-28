//last change: by Hanchuan Peng. 2012-12-30

#ifndef __VN2_PARA_H__
#define __VN2_PARA_H__

//#define SLOT(a) "1"#a
//#define SIGNAL(a) "2"#a

#include "vn.h"

struct PARA_APP2: public PARA_VN
{
    bool is_gsdt;
    bool is_coverage_prune;
    bool is_break_accept;
    int  bkg_thresh;
    double bkg_thresh_k;
    bool rotation;
    bool ini_swc;
    double length_thresh;
    int  cnn_type;
    int  channel;
    double SR_ratio;
    int  b_256cube;
    bool b_RadiusFrom2D; //how to estimate radius of each reconstruction node, from 2D plane (for anisotropic case) or 3D (for isotropic case)
    int b_resample;
    int b_intensity;
    int b_brightfiled;
    bool s2Mode;

    bool b_menu;

    
    QString inimg_file, inmarker_file, outswc_file;
    
    PARA_APP2() 
    {
        is_gsdt = false; //true; change to false by PHC, 2012-10-11. as there is clear GSDT artifacts
        is_coverage_prune = true;
        is_break_accept = false;
        bkg_thresh = 10; //30; change to 10 by PHC 2012-10-11
        bkg_thresh_k = 0.5;
        rotation = false;
        ini_swc = true;
        length_thresh = 5; // 1.0; change to 5 by PHC 2012-10-11
        cnn_type = 2; // default connection type 2
        channel = 0;
        SR_ratio = 3.0/9.0;
        b_256cube = 1; //whether or not preprocessing to downsample to a 256xYxZ cube UINT8 for tracing
        b_RadiusFrom2D = true;
        b_resample = 1;
        b_intensity = 0;
        b_brightfiled = 0;
        b_menu = true;

        inimg_file = "";
        inmarker_file = "";
        outswc_file = "";
    }
    
    bool app2_dialog()
    {
        if (!p4dImage || !p4dImage->valid())
            return false;
        
        int chn_num = p4dImage->getCDim();
        // fetch parameters from dialog
        {
            //set update the dialog
            QDialog * dialog = new QDialog();
            dialog->setWindowTitle("Vaa3D-Neuron3 Auto_tracing Based on APP3 (All-Path-Pruning-v3)");
            QGridLayout * layout = new QGridLayout();
            
            QSpinBox * channel_spinbox = new QSpinBox();
            channel_spinbox->setRange(1,chn_num);
            channel_spinbox->setValue(1);
            QSpinBox * cnntype_spinbox = new QSpinBox();
            cnntype_spinbox->setRange(1,3);
            cnntype_spinbox->setValue(2);
            QSpinBox * bkgthresh_spinbox = new QSpinBox();
            bkgthresh_spinbox->setRange(-1, 255);
            bkgthresh_spinbox->setValue(bkg_thresh);
            QDoubleSpinBox * bkgthreshk_spinbox = new QDoubleSpinBox();
            bkgthreshk_spinbox->setRange(0.0, 2.0);
            bkgthreshk_spinbox->setSingleStep(0.1);
            bkgthreshk_spinbox->setValue(bkg_thresh_k);
            QLineEdit * lenthresh_editor = new QLineEdit(QString("").setNum(length_thresh));
            QLineEdit * srratio_editor = new QLineEdit(QString("").setNum(SR_ratio));
            QCheckBox * isgsdt_checker = new QCheckBox();
            isgsdt_checker->setChecked(is_gsdt);
            QCheckBox * iswb_checker = new QCheckBox();
            iswb_checker->setChecked(is_break_accept);
            QCheckBox * b256_checker = new QCheckBox();
            b256_checker->setChecked(b_256cube);
            QCheckBox * b_radius2Dchecker = new QCheckBox();
            b_radius2Dchecker->setChecked(b_RadiusFrom2D);
            QCheckBox * bresample_Checker = new QCheckBox();
            bresample_Checker->setChecked(b_resample);
            QCheckBox * b_intensity_Checker = new QCheckBox();
            b_intensity_Checker->setChecked(b_intensity);
            QCheckBox * b_brightfiled_Checker = new QCheckBox();
            b_brightfiled_Checker->setChecked(b_brightfiled);
            QCheckBox * rotation_Checker = new QCheckBox();
            rotation_Checker->setChecked(rotation);
            QCheckBox * iniswc_Checker = new QCheckBox();
            iniswc_Checker->setChecked(ini_swc);

            layout->addWidget(new QLabel("color channel"),0,0);
            layout->addWidget(channel_spinbox, 0,1,1,5);
            layout->addWidget(new QLabel("background_threshold \n(if set as -1, \nthen auto-thresholding)"),1,0);
            layout->addWidget(bkgthresh_spinbox, 1,1,1,5);
            layout->addWidget(new QLabel("background_threshold_k \n(0.0~1.0)"),2,0);
            layout->addWidget(bkgthreshk_spinbox, 2,1,1,5);
            
            QHBoxLayout * hbox1 = new QHBoxLayout();
            hbox1->addWidget(new QLabel("auto-downsample"));
            hbox1->addWidget(b256_checker);
            hbox1->addWidget(new QLabel("use GSDT"));
            hbox1->addWidget(isgsdt_checker);
            hbox1->addWidget(new QLabel("allow gap"));
            hbox1->addWidget(iswb_checker);
            hbox1->addWidget(new QLabel("radius from 2D?"));
            hbox1->addWidget(b_radius2Dchecker);

            QHBoxLayout * hbox2 = new QHBoxLayout();
            hbox2->addWidget(new QLabel("auto-resample SWC"));
            hbox2->addWidget(bresample_Checker);
            hbox2->addWidget(new QLabel("high intensity background"));
            hbox2->addWidget(b_intensity_Checker);
            hbox2->addWidget(new QLabel("bright filed"));
            hbox2->addWidget(b_brightfiled_Checker);

            QHBoxLayout * hbox_new = new QHBoxLayout();
            hbox_new->addWidget(new QLabel("rotated augmentation"));
            hbox_new->addWidget(rotation_Checker);
            hbox_new->addWidget(new QLabel("save ini_swc"));
            hbox_new->addWidget(iniswc_Checker);


            layout->addLayout(hbox1,3,0,1,6);
            layout->addLayout(hbox2,4,0,1,6);
            layout->addLayout(hbox_new,5,0,1,6);

            
            layout->addWidget(new QLabel("cnn_type"),6,0);
            layout->addWidget(cnntype_spinbox, 6,1,1,5);
            layout->addWidget(new QLabel("length_thresh"),7,0);
            layout->addWidget(lenthresh_editor, 7,1,1,5);
            layout->addWidget(new QLabel("SR_ratio"),8,0);
            layout->addWidget(srratio_editor, 8,1,1,5);
            
            QHBoxLayout * hbox3 = new QHBoxLayout();
            QPushButton * ok = new QPushButton(" ok ");
            ok->setDefault(true);
            QPushButton * cancel = new QPushButton("cancel");
            hbox3->addWidget(cancel);
            hbox3->addWidget(ok);
            
            layout->addLayout(hbox3,9,0,1,6);
            dialog->setLayout(layout);
            QObject::connect(ok, SIGNAL(clicked()), dialog, SLOT(accept()));
            QObject::connect(cancel, SIGNAL(clicked()), dialog, SLOT(reject()));
            
            //run the dialog
            
            if(dialog->exec() != QDialog::Accepted)
                return false;
            
            //get the dialog return values
            channel = channel_spinbox->value() - 1;
            cnn_type = cnntype_spinbox->value();
            bkg_thresh = bkgthresh_spinbox->value();
            bkg_thresh_k = bkgthreshk_spinbox->value();
            length_thresh = atof(lenthresh_editor->text().toStdString().c_str());
            SR_ratio = atof(srratio_editor->text().toStdString().c_str());
            is_gsdt = isgsdt_checker->isChecked();
            is_break_accept = iswb_checker->isChecked();
            b_256cube = b256_checker->isChecked();
            b_RadiusFrom2D = b_radius2Dchecker->isChecked();
            b_resample = bresample_Checker->isChecked();
            b_intensity = b_intensity_Checker->isChecked();
            b_brightfiled = b_brightfiled_Checker->isChecked();
            rotation = rotation_Checker->isChecked();
            ini_swc = iniswc_Checker->isChecked();
            
            if (dialog) {delete dialog; dialog=0;}
        }
        
        return true;
    }
    
    bool fetch_para_commandline(const V3DPluginArgList &input, V3DPluginArgList &output, V3DPluginCallback2 &callback, QWidget *parent);
};

bool proc_app2(V3DPluginCallback2 &callback, PARA_APP2 &p, const QString & versionStr);

#endif

