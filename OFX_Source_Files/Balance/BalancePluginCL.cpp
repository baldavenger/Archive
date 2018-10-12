#include "BalancePlugin.h"

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

#define kPluginName "Balance"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"White Balance Image: use eyedropper to sample pixel RGB values. If White Balance switch\n" \
"is on then the Red channel is multiplied by the sampled pixel's Green value divided by the\n" \
"sampled pixel's Red value, and the Blue channel is multiplied by the sampled pixel's Green\n" \
"value divided by the sampled pixel's Blue value. If Log Scan switch is also on, then instead\n" \
"the result of subtracting the sampled pixel's Red value from the sampled pixel's Green value\n" \
"is added to the Red channel, the result of subtracting the sampled pixel's Blue value from the \n" \
"sampled pixel's Green value is added to the Blue channel."

#define kPluginIdentifier "OpenFX.Yo.Balance"
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

class ImageScaler : public OFX::ImageProcessor
{
public:
    explicit ImageScaler(OFX::ImageEffect& p_Instance);

    //virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
	void setScales(float p_BalR, float p_BalB, float p_LogBalR, float p_LogBalB, 
		float p_WhiteA, float p_WhiteB);

private:
    OFX::Image* _srcImg;
    float _bal[2];
    float _logbal[2];
    float _whiteF[2];
};

ImageScaler::ImageScaler(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}
/*
extern void RunCudaKernel(int p_Width, int p_Height, float* p_Bal, float* p_LogBal,  
	float* p_White,  const float* p_Input, float* p_Output);

void ImageScaler::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

	RunCudaKernel(width, height, _bal, _logbal, _whiteF, input, output);
}
*/


extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Bal, float* p_LogBal, 
	float* p_White, const float* p_Input, float* p_Output);

void ImageScaler::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());


    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _bal, _logbal, _whiteF, input, output);
}

void ImageScaler::multiThreadProcessImages(OfxRectI p_ProcWindow)
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
                    dstPix[0] = (_whiteF[0]==1.0f) && (_whiteF[1]==1.0f) ? srcPix[0] + _logbal[0] : ((_whiteF[0]==1.0f) ? srcPix[0] * _bal[0] : srcPix[0]);
                    dstPix[1] = srcPix[1];
                    dstPix[2] = (_whiteF[0]==1.0f) && (_whiteF[1]==1.0f) ? srcPix[2] + _logbal[1] : ((_whiteF[0]==1.0f) ? srcPix[0] * _bal[1] : srcPix[2]);
                    dstPix[3] = srcPix[3];
                
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

void ImageScaler::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void ImageScaler::setScales(float p_BalR, float p_BalB, float p_LogBalR, 
	    float p_LogBalB, float p_WhiteA, float p_WhiteB)
{
    _bal[0] = p_BalR;
    _bal[1] = p_BalB;
    _logbal[0] = p_LogBalR;
    _logbal[1] = p_LogBalB;
    _whiteF[0] = p_WhiteA;
    _whiteF[1] = p_WhiteB;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class BalancePlugin : public OFX::ImageEffect
{
public:
    explicit BalancePlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    
    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(ImageScaler &p_ImageScaler, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

    OFX::RGBParam *m_Balance;
    OFX::Double3DParam* m_Rgb;
    OFX::Double3DParam* m_Hsl;
    OFX::BooleanParam* m_White;
    OFX::BooleanParam* m_Log;
	OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
    
    
};

BalancePlugin::BalancePlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_Balance = fetchRGBParam("balance");
    m_Rgb = fetchDouble3DParam("rgbVal");
    m_Hsl = fetchDouble3DParam("hslVal");
    m_White = fetchBooleanParam("whitebalance");
    m_Log = fetchBooleanParam("log");
	m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");
}

void BalancePlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        ImageScaler imageScaler(*this);
        setupAndProcess(imageScaler, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool BalancePlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    
    RGBValues balance;
    m_Balance->getValueAtTime(p_Args.time, balance.r, balance.g, balance.b);
    
    
    float rBalance = balance.r;
    float gBalance = balance.g;
    float bBalance = balance.b;
    

    if ((rBalance == 1.0) && (gBalance == 1.0) && (bBalance == 1.0))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void BalancePlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
    
    if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if(p_ParamName == "button1")
    {
    
    RGBValues balance;
    m_Balance->getValueAtTime(p_Args.time, balance.r, balance.g, balance.b);
    
    float BalanceR = balance.r;
    float BalanceG = balance.g;
    float BalanceB = balance.b;
    
    bool aWhite = m_White->getValueAtTime(p_Args.time);
    bool bWhite = m_Log->getValueAtTime(p_Args.time);

	int White = aWhite ? 1 : 0;
	int Log = bWhite ? 1 : 0;
    
    string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// BalancePlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"\n" \
	"	// switches for whitebalance linear, log\n" \
	"	int whiteA = %d;\n" \
	"	bool p_WhiteA = whiteA == 1;\n" \
	"	int whiteB = %d;\n" \
	"	bool p_WhiteB = whiteB == 1;\n" \
	"	\n" \
	"	// sample pixel RGB values\n" \
	"	float BalanceR = %ff;\n" \
	"	float BalanceG = %ff;\n" \
	"	float BalanceB = %ff;\n" \
	"\n" \
	"	float p_BalR = BalanceG/BalanceR;\n" \
	"	float p_BalB = BalanceG/BalanceB;\n" \
	"\n" \
	"	float p_LogBalR = BalanceG - BalanceR;\n" \
	"	float p_LogBalB = BalanceG - BalanceB;\n" \
	"\n" \
	"	float r = p_WhiteA && p_WhiteB ? p_R + p_LogBalR : p_WhiteA ? p_R * p_BalR : p_R;\n" \
	"	float g = p_G;\n" \
	"	float b = p_WhiteA && p_WhiteB ? p_B + p_LogBalB : p_WhiteA ? p_B * p_BalB : p_B;\n" \
	"\n" \
	"	return make_float3(r, g, b);\n" \
	"}\n", White, Log, BalanceR, BalanceG, BalanceB);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}

	if(p_ParamName == "button2")
    {
    
    RGBValues balance;
    m_Balance->getValueAtTime(p_Args.time, balance.r, balance.g, balance.b);
    
    float BalanceR = balance.r;
    float BalanceG = balance.g;
    float BalanceB = balance.b;
    
    bool aWhite = m_White->getValueAtTime(p_Args.time);
    bool bWhite = m_Log->getValueAtTime(p_Args.time);

	int White = aWhite ? 1 : 0;
	int Log = bWhite ? 1 : 0;
    
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
	" name Balance\n" \
	" xpos -247\n" \
	" ypos -90\n" \
	"}\n" \
	" Input {\n" \
	"  inputs 0\n" \
	"  name Input1\n" \
	"  xpos -268\n" \
	"  ypos -123\n" \
	" }\n" \
	"set N2596e980 [stack 0]\n" \
	" Expression {\n" \
	"  temp_name0 BalanceR\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 BalanceG\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 BalanceB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"r + (BalanceG - BalanceR)\"\n" \
	"  expr1 g\n" \
	"  expr2 \"b + (BalanceG - BalanceB)\"\n" \
	"  name LogScanBalance\n" \
	"  xpos -419\n" \
	"  ypos -69\n" \
	" }\n" \
	"push $N2596e980\n" \
	" Expression {\n" \
	"  temp_name0 BalanceR\n" \
	"  temp_expr0 1.0\n" \
	"  temp_name1 BalanceG\n" \
	"  temp_expr1 1.0\n" \
	"  temp_name2 BalanceB\n" \
	"  temp_expr2 1.0\n" \
	"  expr0 \"r * (BalanceG / BalanceR)\"\n" \
	"  expr1 g\n" \
	"  expr2 \"b * (BalanceG / BalanceB)\"\n" \
	"  name WhiteBalance\n" \
	"  xpos -330\n" \
	"  ypos -71\n" \
	" }\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name Log_switch\n" \
	"  xpos -341\n" \
	"  ypos -5\n" \
	" }\n" \
	"push $N2596e980\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name Balance_switch\n" \
	"  xpos -268\n" \
	"  ypos 34\n" \
	" }\n" \
	" Output {\n" \
	"  name Output1\n" \
	"  xpos -268\n" \
	"  ypos 134\n" \
	" }\n" \
	"end_group\n", BalanceR, BalanceG, BalanceB, Log, White);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
	}	
	}
	}

    if (p_ParamName == "balance")
    {
	 RGBValues balance;
	 m_Balance->getValueAtTime(p_Args.time, balance.r, balance.g, balance.b);
	 
	 m_Rgb->setValue(balance.r, balance.g, balance.b);
	 
	 float Min = std::min(balance.r, std::min(balance.g, balance.b));    
	 float Max = std::max(balance.r, std::max(balance.g, balance.b));    
	 float del_Max = Max - Min;
		
	 float L = (Max + Min) / 2.0f;
	 float S = del_Max == 0.0f ? 0.0f : (L < 0.5f ? del_Max / (Max + Min) : del_Max / (2.0f - Max - Min));

	 float del_R = (((Max - balance.r) / 6.0f) + (del_Max / 2.0f)) / del_Max;
	 float del_G = (((Max - balance.g) / 6.0f) + (del_Max / 2.0f)) / del_Max;
	 float del_B = (((Max - balance.b) / 6.0f) + (del_Max / 2.0f)) / del_Max;

	 float h = del_Max == 0.0f ? 0.0f : (balance.r == Max ? del_B - del_G : (balance.g == Max ? (1.0f / 3.0f) + del_R - del_B : (2.0f / 3.0f) + del_G - del_R));
	 
	 float H = h < 0.0f ? h + 1.0f : (h > 1.0f ? h - 1.0f : h);
	 
	 m_Hsl->setValue(H, S, L);
  		 
    }
}


void BalancePlugin::setupAndProcess(ImageScaler& p_ImageScaler, const OFX::RenderArguments& p_Args)
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

    RGBValues balance;
    m_Balance->getValueAtTime(p_Args.time, balance.r, balance.g, balance.b);
    
    
    float BalanceR = balance.r;
    float BalanceG = balance.g;
    float BalanceB = balance.b;
    
    float rBal = BalanceG/BalanceR;
    float bBal = BalanceG/BalanceB;
    float rlogBal = BalanceG - BalanceR;
    float blogBal = BalanceG - BalanceB;
    
    bool aWhite = m_White->getValueAtTime(p_Args.time);
    bool bWhite = m_Log->getValueAtTime(p_Args.time);

	float  myP1 = aWhite ? 1.0f : 0.0f;
	float  myP2 = bWhite ? 1.0f : 0.0f;
    
    // Set the images
    p_ImageScaler.setDstImg(dst.get());
    p_ImageScaler.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_ImageScaler.setGPURenderArgs(p_Args);

    // Set the render window
    p_ImageScaler.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_ImageScaler.setScales(rBal, bBal, rlogBal, blogBal, myP1, myP2);

    // Call the base class process member, this will call the derived templated process code
    p_ImageScaler.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

BalancePluginFactory::BalancePluginFactory()
    : OFX::PluginFactoryHelper<BalancePluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void BalancePluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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


void BalancePluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
    
    {
        RGBParamDescriptor *param = p_Desc.defineRGBParam("balance");
        param->setLabel("Sample Pixel");
        param->setHint("sample pixel RGB value");
        param->setDefault(1.0, 1.0, 1.0);
        param->setRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
        param->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
        param->setAnimates(true); // can animate
        page->addChild(*param);
    }

	Double3DParamDescriptor* rgbVal = p_Desc.defineDouble3DParam("rgbVal");
    rgbVal->setLabel("RGB Values");
    //pixelVal->setDoubleType(OFX::eDoubleTypeXYAbsolute);
    rgbVal->setDimensionLabels("r", "g", "b");
    rgbVal->setDefault(1.0, 1.0, 1.0);
    //pixelVal->setIncrement(0.0001);
    rgbVal->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
    //pixelVal->setAnimates(true); // can animate
    page->addChild(*rgbVal);
    
    Double3DParamDescriptor* hslVal = p_Desc.defineDouble3DParam("hslVal");
    hslVal->setLabel("HSL Values");
    //pixelVal->setDoubleType(OFX::eDoubleTypeXYAbsolute);
    hslVal->setDimensionLabels("r", "g", "b");
    hslVal->setDefault(0.0, 0.0, 1.0);
    //pixelVal->setIncrement(0.0001);
    hslVal->setDisplayRange(-1.0, -1.0, -1.0, 4.0, 4.0, 4.0);
    //pixelVal->setAnimates(true); // can animate
    page->addChild(*hslVal);

	BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("whitebalance");
    boolParam->setDefault(false);
    boolParam->setHint("white balance image");
    boolParam->setLabel("White Balance");
    page->addChild(*boolParam);
    
    boolParam = p_Desc.defineBooleanParam("log");
    boolParam->setDefault(false);
    boolParam->setHint("performs offset instead of gain");
    boolParam->setLabel("Log Scan");
    page->addChild(*boolParam);
    
    {
    PushButtonParamDescriptor* param = p_Desc.definePushButtonParam("info");
    param->setLabel("Info");
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
	param->setDefault("Balance");
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

ImageEffect* BalancePluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new BalancePlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static BalancePluginFactory balancePlugin;
    p_FactoryArray.push_back(&balancePlugin);
}
