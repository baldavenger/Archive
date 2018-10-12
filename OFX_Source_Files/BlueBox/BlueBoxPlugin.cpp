#include "BlueBoxPlugin.h"

#include <cstring>
#include <cmath>
#include <stdio.h>
using std::string;
#include <string> 
#include <fstream>

#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include "ofxsProcessing.h"
#include "ofxsLog.h"


#ifdef __APPLE__
#define kPluginScript "/Library/Application Support/Blackmagic Design/DaVinci Resolve/LUT"
#elif defined(_WIN32) || defined(__WIN32__) || defined(WIN32) || defined(_WIN64) || defined(__WIN64__) || defined(WIN64)
#define kPluginScript "\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\LUT"
#else
#define kPluginScript "/home/resolve/LUT"
#endif

#define kPluginName "BlueBox"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"BlueBox: Limits the Blue channel so that it is never higher than the Red channel. Combine \n" \
"with the built in Luma Keyer to control blue/cyan cast in the shadows."

#define kPluginIdentifier "OpenFX.Yo.BlueBox"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

namespace {
    struct RGBValues {
        double r,g,b;
        RGBValues(double v) : r(v), g(v), b(v) {}
        RGBValues() : r(0), g(0), b(0) {}
    };
}

class BlueBox : public OFX::ImageProcessor
{
public:
    explicit BlueBox(OFX::ImageEffect& p_Instance);

	virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float p_ScaleR, float p_ScaleG, float p_ScaleB, float p_ScaleA, float p_ScaleD, float p_ScaleE, float p_ScaleO,
		float p_ScaleZ, float p_SwitchA, float p_SwitchB);

private:
    OFX::Image* _srcImg;
    float _scales[8];
	float _switchF[2];
};

BlueBox::BlueBox(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

extern void RunCudaKernel(int p_Width, int p_Height, float* p_BlueBox, float* p_Switch, const float* p_Input, float* p_Output);

void BlueBox::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, _scales, _switchF, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_BlueBox, float* p_Switch, const float* p_Input, float* p_Output);

void BlueBox::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _scales, _switchF, input, output);
}

void BlueBox::multiThreadProcessImages(OfxRectI p_ProcWindow)
{
    for (int y = p_ProcWindow.y1; y < p_ProcWindow.y2; ++y)
    {
        if (_effect.abort()) break;

        float* dstPix = static_cast<float*>(_dstImg->getPixelAddress(p_ProcWindow.x1, y));

        for (int x = p_ProcWindow.x1; x < p_ProcWindow.x2; ++x)
        {
            float* srcPix = static_cast<float*>(_srcImg ? _srcImg->getPixelAddress(x, y) : 0);

            // do we have a source image to scale up
            if (srcPix)
            {
                float l = (srcPix[0] * 0.2126f) + (srcPix[1] * 0.7152f) + (srcPix[2] * 0.0722f);
                float blue = srcPix[2] > srcPix[0] ? srcPix[0] : srcPix[2];
				float L = l - _scales[6];									
				float q = fmin(L, 1.0f);									
				float n = fmax(q, 0.0f);											
																	
				float r = _scales[0];					
				float g = _scales[1];					
				float b = _scales[2];					
				float a = _scales[3];					
				float d = 1.0f / _scales[4];								
				float e = 1.0f / _scales[5];								
				float z = _scales[7];								
									
				float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= n ? 1.0f : (r >= n ? pow((r - n) / (1.0f - g), d) : 0.0f));			
				float k = a == 1.0f ? 0.0f : (a + b <= n ? 1.0f : (a <= n ? pow((n - a) / b, e) : 0.0f));							
				float alpha = k * w;									
				float alphaM = alpha + (1.0f - alpha) * z;				
				float alphaV = (_switchF[1]==1.0f) ? 1.0f - alphaM : alphaM;		
       																		
				dstPix[0] = (_switchF[0] == 1.0f) ? alphaV : srcPix[0];
				dstPix[1] = (_switchF[0] == 1.0f) ? alphaV : srcPix[1];
				dstPix[2] = (_switchF[0] == 1.0f) ? alphaV : srcPix[2] * (1.0f - alphaV) + (blue * alphaV);
				dstPix[3] = (_switchF[0] == 1.0f) ? srcPix[3] : alphaV;
            }
            else
            {
                // no src pixel here, be black and transparent
                for (int c = 0; c < 4; ++c)
                {
                    dstPix[c] = 0;
                }
            }

            // increment the dst pixel
            dstPix += 4;
        }
    }
}

void BlueBox::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void BlueBox::setScales(float p_ScaleR, float p_ScaleG, float p_ScaleB, float p_ScaleA, float p_ScaleD,
	float p_ScaleE, float p_ScaleO, float p_ScaleZ, float p_SwitchA, float p_SwitchB)
{
    _scales[0] = p_ScaleR;
    _scales[1] = p_ScaleG;
    _scales[2] = p_ScaleB;
    _scales[3] = p_ScaleA;
    _scales[4] = p_ScaleD;
    _scales[5] = p_ScaleE;
    _scales[6] = p_ScaleO;
    _scales[7] = p_ScaleZ;
    
    _switchF[0] = p_SwitchA;
    _switchF[1] = p_SwitchB;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class BlueBoxPlugin : public OFX::ImageEffect
{
public:
    explicit BlueBoxPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(BlueBox &p_BlueBox, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

	OFX::RGBParam *m_Sample;
	OFX::DoubleParam* m_ScaleL;
    OFX::DoubleParam* m_ScaleR;
    OFX::DoubleParam* m_ScaleG;
    OFX::DoubleParam* m_ScaleB;
    OFX::DoubleParam* m_ScaleA;
    OFX::DoubleParam* m_ScaleD;
    OFX::DoubleParam* m_ScaleE;
    OFX::DoubleParam* m_ScaleO;
    OFX::DoubleParam* m_ScaleZ;
    OFX::BooleanParam* m_SwitchA;
    OFX::BooleanParam* m_SwitchB;
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
};

BlueBoxPlugin::BlueBoxPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

	m_Sample = fetchRGBParam("sample");
    m_ScaleL = fetchDoubleParam("luma");
    m_ScaleR = fetchDoubleParam("scaleR");
    m_ScaleG = fetchDoubleParam("scaleG");
    m_ScaleB = fetchDoubleParam("scaleB");
    m_ScaleA = fetchDoubleParam("scaleA");
    m_ScaleD = fetchDoubleParam("scaleD");
    m_ScaleE = fetchDoubleParam("scaleE");
    m_ScaleO = fetchDoubleParam("scaleO");
    m_ScaleZ = fetchDoubleParam("scaleZ");
    m_SwitchA = fetchBooleanParam("display");
    m_SwitchB = fetchBooleanParam("invertAlpha");
    m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");

}

void BlueBoxPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        BlueBox BlueBox(*this);
        setupAndProcess(BlueBox, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool BlueBoxPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
   
    double rScale = m_ScaleR->getValueAtTime(p_Args.time);
    double gScale = m_ScaleG->getValueAtTime(p_Args.time);
    double bScale = m_ScaleB->getValueAtTime(p_Args.time);
    double aScale = m_ScaleA->getValueAtTime(p_Args.time);
    double dScale = m_ScaleD->getValueAtTime(p_Args.time);
    double eScale = m_ScaleE->getValueAtTime(p_Args.time);
    double oScale = m_ScaleO->getValueAtTime(p_Args.time);
    double zScale = m_ScaleZ->getValueAtTime(p_Args.time);
    
    if ((rScale == 1.0) && (gScale == 1.0) && (bScale == 0.0) && (aScale == 0.0) && (dScale == 1.0) && (eScale == 1.0) && (oScale == 0.0) && (zScale == 0.0))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void BlueBoxPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    
 	if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if(p_ParamName == "button1")
    {
       
    float high = m_ScaleR->getValueAtTime(p_Args.time);
    float highfade = m_ScaleG->getValueAtTime(p_Args.time);
    float lowfade = m_ScaleB->getValueAtTime(p_Args.time);
    float low = m_ScaleA->getValueAtTime(p_Args.time);
    float highfadecurve = m_ScaleD->getValueAtTime(p_Args.time);
    float lowfadecurve = m_ScaleE->getValueAtTime(p_Args.time);
    float offset = m_ScaleO->getValueAtTime(p_Args.time);
    float mix = m_ScaleZ->getValueAtTime(p_Args.time);
    
    bool Display = m_SwitchA->getValueAtTime(p_Args.time);
    bool Invert = m_SwitchB->getValueAtTime(p_Args.time);

	int display = Display ? 1 : 0;
	int invert = Invert ? 1 : 0;
	
	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// BlueBoxPlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"	// switches for display matte, invert alpha\n" \
	"	int display = %d;\n" \
	"	bool DisplayMatte = display == 1;\n" \
	"	int invert = %d;\n" \
	"	bool InvertAlpha = invert == 1;\n" \
	"\n" \
	"	// matte parameter values\n" \
	"	float high = %ff;\n" \
	"	float highfade = %ff;\n" \
	"	float lowfade = %ff;\n" \
	"	float low = %ff;\n" \
	"	float highfadecurve = %ff;\n" \
	"	float lowfadecurve = %ff;\n" \
	"	float offset = %ff;\n" \
	"	float mix = %ff;\n" \
	"\n" \
	"	float luma = (p_R * 0.2126f) + (p_G * 0.7152f) + (p_B * 0.0722f);\n" \
	"	float blue = p_B > p_R ? p_R : p_B;  \n" \
	"	float L = luma - offset;								\n" \
	"	float q = _fminf(L, 1.0f);									\n" \
	"	float n = _fmaxf(q, 0.0f);\n" \
	"\n" \
	"	float highalpha = high + n == 0.0f ? 0.0f : high - (1.0f - highfade) >= n ? 1.0f : (high >= n ? _powf((high - n) / (1.0f - highfade), highfadecurve) : 0.0f);\n" \
	"	float lowalpha = low + n == 2.0f ? 0.0f : low + lowfade <= n ? 1.0f : (low <= n ? _powf((n - low) / lowfade, lowfadecurve) : 0.0f);\n" \
	"	float alpha = highalpha * lowalpha;\n" \
	"	float alphaM = alpha + (1.0f - alpha) * mix;\n" \
	"	float alphaV = InvertAlpha ? 1.0f - alphaM : alphaM;\n" \
	"	float r = DisplayMatte ? alphaV : p_R;\n" \
	"	float g = DisplayMatte ? alphaV : p_G;\n" \
	"	float b = DisplayMatte ? alphaV : p_B * (1.0f - alphaV) + (blue * alphaV);\n" \
	"\n" \
	"	return make_float3(r, g, b);\n" \
	"}\n", display, invert, high, highfade, lowfade, low, highfadecurve, lowfadecurve, offset, mix);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}

	if(p_ParamName == "button2")
    {
    
    float high = m_ScaleR->getValueAtTime(p_Args.time);
    float highfade = m_ScaleG->getValueAtTime(p_Args.time);
    float lowfade = m_ScaleB->getValueAtTime(p_Args.time);
    float low = m_ScaleA->getValueAtTime(p_Args.time);
    float highfadecurve = m_ScaleD->getValueAtTime(p_Args.time);
    float lowfadecurve = m_ScaleE->getValueAtTime(p_Args.time);
    float offset = m_ScaleO->getValueAtTime(p_Args.time);
    float mix = m_ScaleZ->getValueAtTime(p_Args.time);
    
    bool Display = m_SwitchA->getValueAtTime(p_Args.time);
    bool Invert = m_SwitchB->getValueAtTime(p_Args.time);

	int display = Display ? 1 : 0;
	int invert = Invert ? 1 : 0;
	
	string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".nk to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".nk").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, " Group {\n" \
	" inputs 0\n" \
	" name BlueBox\n" \
	" selected true\n" \
	" xpos -146\n" \
	" ypos 84\n" \
	"}\n" \
	" Input {\n" \
	"  inputs 0\n" \
	"  name Input1\n" \
	"  xpos -432\n" \
	"  ypos 338\n" \
	" }\n" \
	"set N2f74f180 [stack 0]\n" \
	" Expression {\n" \
	"  temp_name0 high\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 highfade\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 highfadecurve\n" \
	"  temp_expr2 %f\n" \
	"  temp_name3 n\n" \
	"  temp_expr3 \"max(min((r * 0.2126 + g * 0.7152 + b * 0.0722) - %f, 1.0), 0.0)\"\n" \
	"  expr0 \"high + n == 0.0 ? 0.0 : high - (1.0 - highfade) >= n ? 1.0 : (high >= n ? pow((high - n) / (1.0 - highfade), highfadecurve) : 0.0)\"\n" \
	"  expr1 0\n" \
	"  expr2 0\n" \
	"  name highalpha\n" \
	"  xpos -524\n" \
	"  ypos 384\n" \
	" }\n" \
	"push $N2f74f180\n" \
	" Expression {\n" \
	"  temp_name0 low\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 lowfade\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 lowfadecurve\n" \
	"  temp_expr2 %f\n" \
	"  temp_name3 n\n" \
	"  temp_expr3 \"max(min((r * 0.2126 + g * 0.7152 + b * 0.0722) - %f, 1.0), 0.0)\"\n" \
	"  expr0 \"low + n == 2.0 ? 0.0 : low + lowfade <= n ? 1.0 : (low <= n ? pow((n - low) / lowfade, lowfadecurve) : 0.0)\"\n" \
	"  expr1 0\n" \
	"  expr2 0\n" \
	"  name lowalpha\n" \
	"  xpos -339\n" \
	"  ypos 379\n" \
	" }\n" \
	" MergeExpression {\n" \
	"  inputs 2\n" \
	"  temp_name0 mix\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 alpha\n" \
	"  temp_expr1 \"Ar * Br\"\n" \
	"  expr0 \"alpha + (1.0 - alpha) * mix\"\n" \
	"  expr1 \"alpha + (1.0 - alpha) * mix\"\n" \
	"  expr2 \"alpha + (1.0 - alpha) * mix\"\n" \
	"  name AlphaM\n" \
	"  xpos -439\n" \
	"  ypos 435\n" \
	" }\n" \
	"set N2f76c930 [stack 0]\n" \
	" Expression {\n" \
	"  expr0 \"1.0 - r\"\n" \
	"  expr1 \"1.0 - r\"\n" \
	"  expr2 \"1.0 - r\"\n" \
	"  name invertAlpha\n" \
	"  xpos -513\n" \
	"  ypos 478\n" \
	" }\n" \
	"push $N2f76c930\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name Invert\n" \
	"  xpos -439\n" \
	"  ypos 523\n" \
	" }\n" \
	"set N2f785320 [stack 0]\n" \
	"push $N2f785320\n" \
	"push $N2f74f180\n" \
	" MergeExpression {\n" \
	"  inputs 2\n" \
	"  temp_name0 blue\n" \
	"  temp_expr0 \"Bb > Br ? Br : Bb\"\n" \
	"  expr0 Br\n" \
	"  expr1 Bg\n" \
	"  expr2 \"Bb * (1.0 - Ar) + (blue * Ar)\"\n" \
	"  name MergeExpression1\n" \
	"  xpos -336\n" \
	"  ypos 562\n" \
	" }\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name Display_matte\n" \
	"  xpos -439\n" \
	"  ypos 618\n" \
	" }\n" \
	" Output {\n" \
	"  name Output1\n" \
	"  xpos -439\n" \
	"  ypos 670\n" \
	" }\n" \
	"end_group\n", high, highfade, highfadecurve, offset, low, lowfade, lowfadecurve, offset, mix, invert, display);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
	}	
	}
	}
	
    if (p_ParamName == "sample")
    {
        
    RGBValues sample;
    m_Sample->getValueAtTime(p_Args.time, sample.r, sample.g, sample.b);
    float luma = (sample.r * 0.2126) + (sample.g * 0.7152) + (sample.b * 0.0722);
    float L1 = luma + 0.1 > 1.0 ? 1.0 : luma + 0.1;
    float L2 = luma - 0.1 < 0.0 ? 0.0 : luma - 0.1;
    float L3 = luma > 0.9 ? 1.0 : 0.95;
    float L4 = luma < 0.1 ? 0.0 : 0.05;
    m_ScaleL->setValue(luma);
    m_ScaleR->setValue(L1);
    m_ScaleG->setValue(L3);
    m_ScaleB->setValue(L4);
    m_ScaleA->setValue(L2);
    
    }
    
     if (p_ParamName == "luma")
    {
    float luma = m_ScaleL->getValueAtTime(p_Args.time);
    float L1 = luma + 0.1 > 1.0 ? 1.0 : luma + 0.1;
    float L2 = luma - 0.1 < 0.0 ? 0.0 : luma - 0.1;
    float L3 = luma > 0.9 ? 1.0 : 0.95;
    float L4 = luma < 0.1 ? 0.0 : 0.05;
    m_ScaleR->setValue(L1);
    m_ScaleG->setValue(L3);
    m_ScaleB->setValue(L4);
    m_ScaleA->setValue(L2);
    
    }
    
}

void BlueBoxPlugin::setupAndProcess(BlueBox& p_BlueBox, const OFX::RenderArguments& p_Args)
{
    // Get the dst image
    std::auto_ptr<OFX::Image> dst(m_DstClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum dstBitDepth = dst->getPixelDepth();
    OFX::PixelComponentEnum dstComponents = dst->getPixelComponents();

    // Get the src image
    std::auto_ptr<OFX::Image> src(m_SrcClip->fetchImage(p_Args.time));
    OFX::BitDepthEnum srcBitDepth = src->getPixelDepth();
    OFX::PixelComponentEnum srcComponents = src->getPixelComponents();

    // Check to see if the bit depth and number of components are the same
    if ((srcBitDepth != dstBitDepth) || (srcComponents != dstComponents))
    {
        OFX::throwSuiteStatusException(kOfxStatErrValue);
    }

    double rScale = m_ScaleR->getValueAtTime(p_Args.time);
    double gScale = m_ScaleG->getValueAtTime(p_Args.time);
    double bScale = m_ScaleB->getValueAtTime(p_Args.time);
    double aScale = m_ScaleA->getValueAtTime(p_Args.time);
    double dScale = m_ScaleD->getValueAtTime(p_Args.time);
    double eScale = m_ScaleE->getValueAtTime(p_Args.time);
    double oScale = m_ScaleO->getValueAtTime(p_Args.time);
    double zScale = m_ScaleZ->getValueAtTime(p_Args.time);
    
    bool aSwitch = m_SwitchA->getValueAtTime(p_Args.time);
    bool bSwitch = m_SwitchB->getValueAtTime(p_Args.time);

	float aSwitchF = (aSwitch) ? 1.0f : 0.0f;
	float bSwitchF = (bSwitch) ? 1.0f : 0.0f;

    // Set the images
    p_BlueBox.setDstImg(dst.get());
    p_BlueBox.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_BlueBox.setGPURenderArgs(p_Args);

    // Set the render window
    p_BlueBox.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_BlueBox.setScales(rScale, gScale, bScale, aScale, dScale, eScale, oScale, zScale, aSwitchF, bSwitchF);

    // Call the base class process member, this will call the derived templated process code
    p_BlueBox.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

BlueBoxPluginFactory::BlueBoxPluginFactory()
    : OFX::PluginFactoryHelper<BlueBoxPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void BlueBoxPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
{
    // Basic labels
    p_Desc.setLabels(kPluginName, kPluginName, kPluginName);
    p_Desc.setPluginGrouping(kPluginGrouping);
    p_Desc.setPluginDescription(kPluginDescription);

    // Add the supported contexts, only filter at the moment
    p_Desc.addSupportedContext(eContextFilter);
    p_Desc.addSupportedContext(eContextGeneral);

    // Add supported pixel depths
    p_Desc.addSupportedBitDepth(eBitDepthFloat);

    // Set a few flags
    p_Desc.setSingleInstance(false);
    p_Desc.setHostFrameThreading(false);
    p_Desc.setSupportsMultiResolution(kSupportsMultiResolution);
    p_Desc.setSupportsTiles(kSupportsTiles);
    p_Desc.setTemporalClipAccess(false);
    p_Desc.setRenderTwiceAlways(false);
    p_Desc.setSupportsMultipleClipPARs(kSupportsMultipleClipPARs);

    // Setup OpenCL and CUDA render capability flags
    p_Desc.setSupportsOpenCLRender(true);
    p_Desc.setSupportsCudaRender(true);
    
}

static DoubleParamDescriptor* defineScaleParam(OFX::ImageEffectDescriptor& p_Desc, const std::string& p_Name, const std::string& p_Label,
                                               const std::string& p_Hint, GroupParamDescriptor* p_Parent)
{
    DoubleParamDescriptor* param = p_Desc.defineDoubleParam(p_Name);
    param->setLabels(p_Label, p_Label, p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    param->setDoubleType(eDoubleTypeScale);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void BlueBoxPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
{
    // Source clip only in the filter context
    // Create the mandated source clip
    ClipDescriptor* srcClip = p_Desc.defineClip(kOfxImageEffectSimpleSourceClipName);
    srcClip->addSupportedComponent(ePixelComponentRGBA);
    srcClip->setTemporalClipAccess(false);
    srcClip->setSupportsTiles(kSupportsTiles);
    srcClip->setIsMask(false);

    // Create the mandated output clip
    ClipDescriptor* dstClip = p_Desc.defineClip(kOfxImageEffectOutputClipName);
    dstClip->addSupportedComponent(ePixelComponentRGBA);
    dstClip->addSupportedComponent(ePixelComponentAlpha);
    dstClip->setSupportsTiles(kSupportsTiles);

    // Make some pages and to things in
    PageParamDescriptor* page = p_Desc.definePageParam("Controls");

    // Group param to group the scales
    GroupParamDescriptor* componentScalesGroup = p_Desc.defineGroupParam("alphaChannel");
    componentScalesGroup->setHint("Pull Luma Key");
    componentScalesGroup->setLabels("BlueBox", "BlueBox", "BlueBox");

    
    // Add a boolean to enable the component scale
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("display");
    boolParam->setDefault(false);
    boolParam->setHint("Displays Matte on RGB Channels");
    boolParam->setLabel("Display Matte");
    boolParam->setParent(*componentScalesGroup);
    page->addChild(*boolParam);
    
    boolParam = p_Desc.defineBooleanParam("invertAlpha");
    boolParam->setDefault(false);
    boolParam->setHint("Inverts the Alpha Channel");
    boolParam->setLabel("Invert");
    boolParam->setParent(*componentScalesGroup);
    page->addChild(*boolParam);

	RGBParamDescriptor *rgbparam = p_Desc.defineRGBParam("sample");
	rgbparam->setLabel("Luma Sample");
	rgbparam->setHint("click on pixel");
	rgbparam->setDefault(1.0, 1.0, 1.0);
	rgbparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
	rgbparam->setAnimates(true); // can animate
	rgbparam->setParent(*componentScalesGroup);
	page->addChild(*rgbparam);
    
   
	DoubleParamDescriptor* param = defineScaleParam(p_Desc, "luma", "luma", "luma from sample", componentScalesGroup);
	param->setDefault(1.0);
	param->setRange(0, 1.0);
	param->setIncrement(0.001);
	param->setDisplayRange(0, 1.0);
	param->setParent(*componentScalesGroup);
	page->addChild(*param);
    

    // Make the component params
    param = defineScaleParam(p_Desc, "scaleR", "high", "High limit of alpha channel", componentScalesGroup);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleG", "high fade", "Roll-off between high limit", componentScalesGroup);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleB", "low fade", "Roll-off between low limit", componentScalesGroup);
    param->setDefault(0.0);
    param->setRange(0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleA", "low", "Low limit of alpha channel", componentScalesGroup);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleD", "high fade curve", "Easy out / Easy in", componentScalesGroup);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.2, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleE", "low fade curve", "Easy out / Easy in", componentScalesGroup);
    param->setDefault(1.0);
    param->setRange(0.2, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.2, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleO", "offset", "Offsets the alpha range", componentScalesGroup);
    param->setDefault(0.0);
    param->setRange(-1.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-1.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleZ", "mix", "Blends new alpha with original alpha", componentScalesGroup);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("info");
    param->setLabel("Info");
    param->setParent(*componentScalesGroup);
    page->addChild(*param);
    }
    
    {    
    GroupParamDescriptor* script = p_Desc.defineGroupParam("Script Export");
    script->setOpen(false);
    script->setHint("export DCTL and Nuke script");
      if (page) {
            page->addChild(*script);
            }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button1");
    param->setLabel("Export DCTL");
    param->setHint("create DCTL version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("button2");
    param->setLabel("Export Nuke script");
    param->setHint("create NUKE version");
    param->setParent(*script);
    page->addChild(*param);
    }
    {
	StringParamDescriptor* param = p_Desc.defineStringParam("name");
	param->setLabel("Name");
	param->setHint("overwrites if the same");
	param->setDefault("BlueBox");
	param->setParent(*script);
	page->addChild(*param);
	}
	{
	StringParamDescriptor* param = p_Desc.defineStringParam("path");
	param->setLabel("Directory");
	param->setHint("make sure it's the absolute path");
	param->setStringType(eStringTypeFilePath);
	param->setDefault(kPluginScript);
	param->setFilePathExists(false);
	param->setParent(*script);
	page->addChild(*param);
	}
	}        
    
}

ImageEffect* BlueBoxPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new BlueBoxPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static BlueBoxPluginFactory BlueBoxPlugin;
    p_FactoryArray.push_back(&BlueBoxPlugin);
}
