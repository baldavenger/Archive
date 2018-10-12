#include "HueKeyPlugin.h"

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

#define kPluginName "HueKey"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"HueKeyer: Hue based keyer. Use eyedropper to isolate specific range and finetune with the \n" \
"controls."

#define kPluginIdentifier "OpenFX.Yo.HueKey"
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

 float RGBtoHUE(float R, float G, float B)
{

	float min = fmin(R, fmin(G, B));    
	float Max = fmax(R, fmax(G, B));    
	float del_Max = Max - min;
	
	float del_R = (((Max - R) / 6.0) + (del_Max / 2.0)) / del_Max;
    float del_G = (((Max - G) / 6.0) + (del_Max / 2.0)) / del_Max;
    float del_B = (((Max - B) / 6.0) + (del_Max / 2.0)) / del_Max;
   
    float h = del_Max == 0.0 ? 0.0 : (R == Max ? del_B - del_G : (G == Max ? (1.0 / 3.0) + del_R - del_B : (2.0 / 3.0) + del_G - del_R));

    float Hh = h < 0.0 ? h + 1.0 : (h > 1.0 ? h - 1.0 : h);
    return Hh;
      
}

class Hue : public OFX::ImageProcessor
{
public:
    explicit Hue(OFX::ImageEffect& p_Instance);

    //virtual void processImagesCUDA();
    //virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);
    
    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float p_ScaleR, float p_ScaleG, float p_ScaleB, float p_ScaleA,
		float p_ScaleD, float p_ScaleE, float p_ScaleO, float p_ScaleZ, float p_SwitchA, float p_SwitchB);

private:
    OFX::Image* _srcImg;
    float _scales[8];
	float _switch[2];
};

Hue::Hue(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}
/*
extern void RunCudaKernel(int p_Width, int p_Height, float* p_Hue, float* p_Switch, const float* p_Input, float* p_Output);

void Hue::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, _scales, _switch, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Hue, float* p_Switch, const float* p_Input, float* p_Output);

void Hue::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _scales, _switch, input, output);
}
*/
void Hue::multiThreadProcessImages(OfxRectI p_ProcWindow)
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
                float h = RGBtoHUE(srcPix[0], srcPix[1], srcPix[2]);											
																	
				float r = _scales[0];					
				float g = _scales[1];					
				float b = _scales[2];					
				float a = _scales[3];					
				float d = 1.0f / _scales[4];								
				float e = 1.0f / _scales[5];								
				float o = _scales[6] * -1.0f;
				float z = _scales[7];
				
				float H = h == 0.0f ? 0.0f : (h + o > 1.0f ? h + o - 1.0f : (h + o < 0.0f ? 1.0f + h + o : h + o));								
									
				float w = r == 0.0f ? 0.0f : (r - (1.0f - g) >= H ? 1.0f : (r >= H ? pow((r - H) / (1.0f - g), d) : 0.0f));			
				float k = a == 1.0f ? 0.0f : (a + b <= H ? 1.0f : (a <= H ? pow((H - a) / b, e) : 0.0f));							
				float alpha = k * w;									
				float alphaM = alpha + (1.0f - alpha) * z;				
				float alphaV = (_switch[1] == 1.0f) ? 1.0f - alphaM : alphaM;		
       																		
				dstPix[0] = (_switch[0] == 1.0f) ? alphaV : srcPix[0];
				dstPix[1] = (_switch[0] == 1.0f) ? alphaV : srcPix[1];
				dstPix[2] = (_switch[0] == 1.0f) ? alphaV : srcPix[2];
				dstPix[3] = (_switch[0] == 1.0f) ? srcPix[3] : alphaV;
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


void Hue::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void Hue::setScales(float p_ScaleR, float p_ScaleG, float p_ScaleB, float p_ScaleA,
	float p_ScaleD, float p_ScaleE, float p_ScaleO, float p_ScaleZ, float p_SwitchA, float p_SwitchB)
{
    _scales[0] = p_ScaleR;
    _scales[1] = p_ScaleG;
    _scales[2] = p_ScaleB;
    _scales[3] = p_ScaleA;
    _scales[4] = p_ScaleD;
    _scales[5] = p_ScaleE;
    _scales[6] = p_ScaleO;
    _scales[7] = p_ScaleZ;
    
    _switch[0] = p_SwitchA;
    _switch[1] = p_SwitchB;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class HueKeyPlugin : public OFX::ImageEffect
{
public:
    explicit HueKeyPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);
    
    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);

    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(Hue &p_Hue, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::RGBParam *m_Sample;
	OFX::DoubleParam* m_ScaleH;
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

HueKeyPlugin::HueKeyPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_Sample = fetchRGBParam("sample");
    m_ScaleH = fetchDoubleParam("Hue");
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

void HueKeyPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        Hue hue(*this);
        setupAndProcess(hue, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool HueKeyPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
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

void HueKeyPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
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
    	
	fprintf (pFile, "// HueKeyPlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"\n" \
	"	// switches for display matte, invert alpha\n" \
	"	int display = %d;\n" \
	"	bool DisplayMatte = display == 1;\n" \
	"	int invert = %d;\n" \
	"	bool InvertAlpha = invert == 1;\n" \
	"\n" \
	"	float Mx = _fmaxf(p_R, _fmaxf(p_G, p_B));\n" \
	"	float mn = _fminf(p_R, _fminf(p_G, p_B));\n" \
	"	float del_Max = Mx - mn;\n" \
	"\n" \
	"	float del_R = (((Mx - p_R) / 6.0f) + (del_Max / 2.0f)) / del_Max;\n" \
	"	float del_G = (((Mx - p_G) / 6.0f) + (del_Max / 2.0f)) / del_Max;\n" \
	"	float del_B = (((Mx - p_B) / 6.0f) + (del_Max / 2.0f)) / del_Max;\n" \
	"\n" \
	"	float h = del_Max == 0.0f ? 0.0f : (p_R == Mx ? del_B - del_G : (p_G == Mx ? \n" \
	"	(1.0f / 3.0f) + del_R - del_B : (2.0f / 3.0f) + del_G - del_R));\n" \
	"\n" \
	"	// matte parameter values\n" \
	"	float high = %ff;\n" \
	"	float highfade = %ff;\n" \
	"	float lowfade = %ff;\n" \
	"	float low = %ff;\n" \
	"	float highfadecurve = %ff;\n" \
	"	float lowfadecurve = %ff;\n" \
	"	float offset = %ff * -1.0f;\n" \
	"	float mix = %ff;\n" \
	"\n" \
	"	float Hue = h == 0.0f ? 0.0f : (h + offset > 1.0f ? h + offset - 1.0f : (h + offset < 0.0f ? 1.0f + h + offset : h + offset));\n" \
	"	float highalpha = high + Hue == 0.0f ? 0.0f : high - (1.0f - highfade) >= Hue ? 1.0f : (high >= Hue ? _powf((high - Hue) / (1.0f - highfade), highfadecurve) : 0.0f);\n" \
	"	float lowalpha = low + Hue == 2.0f ? 0.0f : low + lowfade <= Hue ? 1.0f : (low <= Hue ? _powf((Hue - low) / lowfade, lowfadecurve) : 0.0f);\n" \
	"	float alpha = highalpha * lowalpha;\n" \
	"	float alphaM = alpha + (1.0f - alpha) * mix;\n" \
	"	float alphaV = InvertAlpha ? 1.0f - alphaM : alphaM;\n" \
	"\n" \
	"	float r = DisplayMatte ? alphaV : p_R;\n" \
	"	float g = DisplayMatte ? alphaV : p_G;\n" \
	"	float b = DisplayMatte ? alphaV : p_B;\n" \
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
    	
	fprintf (pFile, "Group {\n" \
	" inputs 0\n" \
	" name HueKey\n" \
	" selected true\n" \
	" xpos -146\n" \
	" ypos 84\n" \
	"}\n" \
	" Input {\n" \
	"  inputs 0\n" \
	"  name Input1\n" \
	"  xpos -386\n" \
	"  ypos 283\n" \
	" }\n" \
	"set N1d300e80 [stack 0]\n" \
	" Colorspace {\n" \
	"  colorspace_in sRGB\n" \
	"  colorspace_out HSL\n" \
	"  name RGB_to_HSL\n" \
	"  xpos -496\n" \
	"  ypos 316\n" \
	" }\n" \
	"set N227306a0 [stack 0]\n" \
	" Expression {\n" \
	"  temp_name0 high\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 highfade\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 offset\n" \
	"  temp_expr2 %f\n" \
	"  temp_name3 hue\n" \
	"  temp_expr3 \"r == 0.0 ? 0.0 : (r + offset > 1.0 ? r + offset - 1.0 : (r + offset < 0.0 ? 1.0 + r + offset : r + offset))\"\n" \
	"  expr0 \"high + n == 0.0 ? 0.0 : high - (1.0 - highfade) >= n ? 1.0 : (high >= n ? pow((high - n) / (1.0 - highfade), %ff) : 0.0)\"\n" \
	"  expr1 0\n" \
	"  expr2 0\n" \
	"  name highalpha\n" \
	"  xpos -537\n" \
	"  ypos 386\n" \
	" }\n" \
	"push $N227306a0\n" \
	" Expression {\n" \
	"  temp_name0 low\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 lowfade\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 offset\n" \
	"  temp_expr2 %f\n" \
	"  temp_name3 n\n" \
	"  temp_expr3 \"r == 0.0 ? 0.0 : (r + offset > 1.0 ? r + offset - 1.0 : (r + offset < 0.0 ? 1.0 + r + offset : r + offset))\"\n" \
	"  expr0 \"low + n == 2.0 ? 0.0 : low + lowfade <= n ? 1.0 : (low <= n ? pow((n - low) / lowfade, %ff) : 0.0)\"\n" \
	"  expr1 0\n" \
	"  expr2 0\n" \
	"  name lowalpha\n" \
	"  xpos -450\n" \
	"  ypos 356\n" \
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
	"  xpos -457\n" \
	"  ypos 417\n" \
	" }\n" \
	"set N1d3c4a50 [stack 0]\n" \
	" Expression {\n" \
	"  expr0 \"1.0 - r\"\n" \
	"  expr1 \"1.0 - r\"\n" \
	"  expr2 \"1.0 - r\"\n" \
	"  name invertAlpha\n" \
	"  xpos -520\n" \
	"  ypos 461\n" \
	" }\n" \
	"push $N1d3c4a50\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name Invert\n" \
	"  xpos -457\n" \
	"  ypos 501\n" \
	" }\n" \
	"push $N1d300e80\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name Display_matte\n" \
	"  xpos -386\n" \
	"  ypos 546\n" \
	" }\n" \
	" Output {\n" \
	"  name Output1\n" \
	"  xpos -386\n" \
	"  ypos 586\n" \
	" }\n" \
	"end_group\n", high, highfade, offset, highfadecurve, low, lowfade, offset, lowfadecurve, mix, invert, display);
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
    float hue = RGBtoHUE(sample.r, sample.g, sample.b);
    float h1 = hue == 0.0 ? 0.0 : (hue + 0.1 > 1.0 ? 1.0 : hue + 0.1);
    float h2 = hue - 0.1 < 0.0001 ? 0.0001 : hue - 0.1;
    float h3 = hue > 0.9 ? 1.0 : 0.95;
    float h4 = hue < 0.1 ? 0.0 : 0.05;
    m_ScaleH->setValue(hue);
    m_ScaleR->setValue(h1);
    m_ScaleG->setValue(h3);
    m_ScaleB->setValue(h4);
    m_ScaleA->setValue(h2);
    
    }
    
    if (p_ParamName == "Hue")
    {
    float hue = m_ScaleH->getValueAtTime(p_Args.time);
    float h1 = hue + 0.1 > 1.0 ? 1.0 : hue + 0.1;
    float h2 = hue - 0.1 < 0.0001 ? 0.0001 : hue - 0.1;
    float h3 = hue > 0.9 ? 1.0 : 0.95;
    float h4 = hue < 0.1 ? 0.0 : 0.05;
    m_ScaleR->setValue(h1);
    m_ScaleG->setValue(h3);
    m_ScaleB->setValue(h4);
    m_ScaleA->setValue(h2);
    
    
    }
    
    if (p_ParamName == "scaleR")
    {
    float l = m_ScaleA->getValueAtTime(p_Args.time);
    float L = l == 0.0 ? 0.0001 : l;
    m_ScaleA->setValue(L);
    
    }
    
    if (p_ParamName == "scaleG")
    {
    float l = m_ScaleA->getValueAtTime(p_Args.time);
    float L = l == 0.0 ? 0.0001 : l;
    m_ScaleA->setValue(L);
    
    }
    
}


void HueKeyPlugin::setupAndProcess(Hue& p_Hue, const OFX::RenderArguments& p_Args)
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

	float aSwitchF = aSwitch ? 1.0f : 0.0f;
	float bSwitchF = bSwitch ? 1.0f : 0.0f;
	
    // Set the images
    p_Hue.setDstImg(dst.get());
    p_Hue.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    //p_Hue.setGPURenderArgs(p_Args);

    // Set the render window
    p_Hue.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_Hue.setScales(rScale, gScale, bScale, aScale, dScale, eScale, oScale, zScale, aSwitchF, bSwitchF);

    // Call the base class process member, this will call the derived templated process code
    p_Hue.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

HueKeyPluginFactory::HueKeyPluginFactory()
    : OFX::PluginFactoryHelper<HueKeyPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void HueKeyPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
    //p_Desc.setSupportsOpenCLRender(true);
    //p_Desc.setSupportsCudaRender(true);
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


void HueKeyPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
    componentScalesGroup->setHint("Pull Hue Key");
    componentScalesGroup->setLabels("Hue Keyer", "Hue Keyer", "Hue Keyer");

    
    // Add a boolean to enable the component scale
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("display");
    boolParam->setDefault(false);
    boolParam->setHint("Displays Alpha on RGB Channels");
    boolParam->setLabels("Display Matte", "Display Matte", "Display Matte");
    boolParam->setParent(*componentScalesGroup);
    page->addChild(*boolParam);
    
    boolParam = p_Desc.defineBooleanParam("invertAlpha");
    boolParam->setDefault(false);
    boolParam->setHint("Inverts the Alpha Channel");
    boolParam->setLabels("Invert", "Invert", "Invert");
    boolParam->setParent(*componentScalesGroup);
    page->addChild(*boolParam);
    
    RGBParamDescriptor *rgbparam = p_Desc.defineRGBParam("sample");
	rgbparam->setLabel("Hue Sample");
	rgbparam->setHint("click on pixel");
	rgbparam->setDefault(1.0, 1.0, 1.0);
	rgbparam->setDisplayRange(0.0, 0.0, 0.0, 4.0, 4.0, 4.0);
	rgbparam->setAnimates(true); // can animate
	rgbparam->setParent(*componentScalesGroup);
	page->addChild(*rgbparam);
    
   
	DoubleParamDescriptor* param = defineScaleParam(p_Desc, "Hue", "Hue", "hue from sample", componentScalesGroup);
	param->setDefault(1.0);
	param->setRange(0.0, 1.0);
	param->setIncrement(0.001);
	param->setDisplayRange(0.0, 1.0);
	param->setParent(*componentScalesGroup);
	page->addChild(*param);

    
    param = defineScaleParam(p_Desc, "scaleR", "high", "High limit of alpha channel", componentScalesGroup);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleG", "high fade", "Roll-off between high limit", componentScalesGroup);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleB", "low fade", "Roll-off between low limit", componentScalesGroup);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);

    param = defineScaleParam(p_Desc, "scaleA", "low", "Low limit of alpha channel", componentScalesGroup);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
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
    
    param = defineScaleParam(p_Desc, "scaleO", "rotate", "Rotates the alpha range", componentScalesGroup);
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
	param->setDefault("HueKey");
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

ImageEffect* HueKeyPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new HueKeyPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static HueKeyPluginFactory HueKeyPlugin;
    p_FactoryArray.push_back(&HueKeyPlugin);
}
