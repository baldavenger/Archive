#include "LiftGammaGainPlugin.h"

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

#define kPluginName "LiftGammaGain"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"LiftGammaGain: Lift, Gamma, Gain, and Offset controls. Order of operations are Gain, Lift, \n" \
"Offset, Gamma."

#define kPluginIdentifier "OpenFX.Yo.LiftGammaGain"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 0

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

class ImageScaler : public OFX::ImageProcessor
{
public:
    explicit ImageScaler(OFX::ImageEffect& p_Instance);

    //virtual void processImagesCUDA();
    //virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float p_ScaleL, float p_ScaleG, float p_ScaleGG, float p_ScaleO);

private:
    OFX::Image* _srcImg;
    float _scales[4];
};

ImageScaler::ImageScaler(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}
/*
extern void RunCudaKernel(int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);

void ImageScaler::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, _scales, input, output);
}

extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Gain, const float* p_Input, float* p_Output);

void ImageScaler::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _scales, input, output);
}
*/
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
                
                    dstPix[0] = pow((srcPix[0] * _scales[2]) + (_scales[0] * (1.0 - (srcPix[0] * _scales[2]))) + _scales[3], 1.0/_scales[1]);
                    dstPix[1] = pow((srcPix[1] * _scales[2]) + (_scales[0] * (1.0 - (srcPix[1] * _scales[2]))) + _scales[3], 1.0/_scales[1]);
                    dstPix[2] = pow((srcPix[2] * _scales[2]) + (_scales[0] * (1.0 - (srcPix[2] * _scales[2]))) + _scales[3], 1.0/_scales[1]);
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

void ImageScaler::setScales(float p_ScaleL, float p_ScaleG, float p_ScaleGG, float p_ScaleO)
{
    _scales[0] = p_ScaleL;
    _scales[1] = p_ScaleG;
    _scales[2] = p_ScaleGG;
    _scales[3] = p_ScaleO;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class LiftGammaGainPlugin : public OFX::ImageEffect
{
public:
    explicit LiftGammaGainPlugin(OfxImageEffectHandle p_Handle);

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

    OFX::DoubleParam* m_ScaleL;
    OFX::DoubleParam* m_ScaleG;
    OFX::DoubleParam* m_ScaleGG;
    OFX::DoubleParam* m_ScaleO;
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
};

LiftGammaGainPlugin::LiftGammaGainPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

    m_ScaleL = fetchDoubleParam("scaleL");
    m_ScaleG = fetchDoubleParam("scaleG");
    m_ScaleGG = fetchDoubleParam("scaleGG");
    m_ScaleO = fetchDoubleParam("scaleO");
    m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");
    
}

void LiftGammaGainPlugin::render(const OFX::RenderArguments& p_Args)
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

bool LiftGammaGainPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{
    
    double lScale = m_ScaleL->getValueAtTime(p_Args.time);
    double gScale = m_ScaleG->getValueAtTime(p_Args.time);
    double ggScale = m_ScaleGG->getValueAtTime(p_Args.time);
    double oScale = m_ScaleO->getValueAtTime(p_Args.time);
    

    if ((lScale == 1.0) && (gScale == 1.0) && (ggScale == 1.0) && (oScale == 0.0))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void LiftGammaGainPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
 
 	if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if(p_ParamName == "button1")
    {
       
    float lift = m_ScaleL->getValueAtTime(p_Args.time);
    float gamma = m_ScaleG->getValueAtTime(p_Args.time);
    float gain = m_ScaleGG->getValueAtTime(p_Args.time);
    float offset = m_ScaleO->getValueAtTime(p_Args.time);
    
    string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// LiftGammaGainPlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"    	\n" \
	"	// Lift, Gamma, Gain, Offset\n" \
	"	float Lift = %ff;\n" \
	"	float Gamma = %ff;\n" \
	"	float Gain = %ff;\n" \
	"	float Offset = %ff;\n" \
	"\n" \
	"	const float r = _powf(p_R * Gain + Lift * (1.0f - p_R * Gain) + Offset, 1.0f/Gamma);\n" \
	"	const float g = _powf(p_G * Gain + Lift * (1.0f - p_G * Gain) + Offset, 1.0f/Gamma);\n" \
	"	const float b = _powf(p_B * Gain + Lift * (1.0f - p_B * Gain) + Offset, 1.0f/Gamma);\n" \
	"\n" \
	"	return make_float3(r, g, b);\n" \
	"}\n", lift, gamma, gain, offset);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}

	if(p_ParamName == "button2")
    {

	float lift = m_ScaleL->getValueAtTime(p_Args.time);
    float gamma = m_ScaleG->getValueAtTime(p_Args.time);
    float gain = m_ScaleGG->getValueAtTime(p_Args.time);
    float offset = m_ScaleO->getValueAtTime(p_Args.time);
    
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
	" name LiftGammaGain\n" \
	" xpos -84\n" \
	" ypos -24\n" \
	"}\n" \
	" Input {\n" \
	"  inputs 0\n" \
	"  name Input1\n" \
	"  xpos -40\n" \
	"  ypos -50\n" \
	" }\n" \
	" Expression {\n" \
	"  temp_name0 lift\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 gamma\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 gain\n" \
	"  temp_expr2 %f\n" \
	"  temp_name3 offset\n" \
	"  temp_expr3 %f\n" \
	"  expr0 \"pow(r * gain + lift * (1.0 - r * gain) + offset, 1.0 / gamma)\"\n" \
	"  expr1 \"pow(g * gain + lift * (1.0 - g * gain) + offset, 1.0 / gamma)\"\n" \
	"  expr2 \"pow(b * gain + lift * (1.0 - b * gain) + offset, 1.0 / gamma)\"\n" \
	"  name liftgammagain\n" \
	"  xpos -40\n" \
	"  ypos -10\n" \
	" }\n" \
	" Output {\n" \
	"  name Output1\n" \
	"  xpos -40\n" \
	"  ypos 90\n" \
	" }\n" \
	"end_group\n", lift, gamma, gain, offset);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
	}	
	}
	}
}

void LiftGammaGainPlugin::setupAndProcess(ImageScaler& p_ImageScaler, const OFX::RenderArguments& p_Args)
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

    double lScale = m_ScaleL->getValueAtTime(p_Args.time);
    double gScale = m_ScaleG->getValueAtTime(p_Args.time);
    double ggScale = m_ScaleGG->getValueAtTime(p_Args.time);
    double oScale = m_ScaleO->getValueAtTime(p_Args.time);

    // Set the images
    p_ImageScaler.setDstImg(dst.get());
    p_ImageScaler.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    //p_ImageScaler.setGPURenderArgs(p_Args);

    // Set the render window
    p_ImageScaler.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_ImageScaler.setScales(lScale, gScale, ggScale, oScale);

    // Call the base class process member, this will call the derived templated process code
    p_ImageScaler.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

LiftGammaGainPluginFactory::LiftGammaGainPluginFactory()
    : OFX::PluginFactoryHelper<LiftGammaGainPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void LiftGammaGainPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
	param->setLabel(p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    //param->setDefault(1.0);
    //param->setRange(-10.0, 10.0);
    //param->setIncrement(0.001);
    //param->setDisplayRange(-5.0, 5.0);
    //param->setDoubleType(eDoubleTypeScale);

    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void LiftGammaGainPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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

    // Make the four component params
    
    
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scaleL", "Lift", "L from the LGG", 0);
    param->setDefault(0.0);
    param->setRange(-2.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-2.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleG", "Gamma", "G from the LGG", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleGG", "Gain", "Double G from the LGG", 0);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleO", "Offset", "Bonus O", 0);
    param->setDefault(0.0);
    param->setRange(-60.0, 20.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-60.0, 20.0);
    page->addChild(*param);
    
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
	param->setDefault("LiftGammaGain");
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

ImageEffect* LiftGammaGainPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new LiftGammaGainPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static LiftGammaGainPluginFactory liftgammagainPlugin;
    p_FactoryArray.push_back(&liftgammagainPlugin);
}
