#include "TwelveWayPlugin.h"

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

#define kPluginName "TwelveWay"
#define kPluginGrouping "OpenFX Yo"
#define kPluginDescription \
"------------------------------------------------------------------------------------------------------------------ \n" \
"TwelveWay: 12 Way Colour Corrector, allowing for Overall, Shadow, Mid-tones, and Highlights \n" \
"control."

#define kPluginIdentifier "OpenFX.Yo.TwelveWay"
#define kPluginVersionMajor 2
#define kPluginVersionMinor 1

#define kSupportsTiles false
#define kSupportsMultiResolution false
#define kSupportsMultipleClipPARs false

////////////////////////////////////////////////////////////////////////////////

class TwelveWay : public OFX::ImageProcessor
{
public:
    explicit TwelveWay(OFX::ImageEffect& p_Instance);

	//virtual void processImagesCUDA();
    virtual void processImagesOpenCL();
    virtual void multiThreadProcessImages(OfxRectI p_ProcWindow);

    void setSrcImg(OFX::Image* p_SrcImg);
    void setScales(float p_SwitchO, float p_SwitchS, float p_SwitchM, float p_SwitchH, float p_ScaleOL, 
	float p_ScaleOG, float p_ScaleOGG, 
    float p_ScaleSL, float p_ScaleSG, float p_ScaleSGG, float p_ScaleSA, float p_ScaleSB, 
    float p_ScaleML, float p_ScaleMG,  float p_ScaleMGG, float p_ScaleMA, float p_ScaleMB, 
    float p_ScaleHL, float p_ScaleHG,  float p_ScaleHGG, float p_ScaleHA, float p_ScaleHB);     

private:
    OFX::Image* _srcImg;
    float _switch[4];
    float _scales[18];
    
};

TwelveWay::TwelveWay(OFX::ImageEffect& p_Instance)
    : OFX::ImageProcessor(p_Instance)
{
}

/*
extern void RunCudaKernel(int p_Width, int p_Height, float* p_Switch, float* p_Gain, const float* p_Input, float* p_Output);

void TwelveWay::processImagesCUDA()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunCudaKernel(width, height, _switch, _scales, input, output);
}
*/
extern void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float* p_Switch, float* p_Gain, const float* p_Input, float* p_Output);

void TwelveWay::processImagesOpenCL()
{
    const OfxRectI& bounds = _srcImg->getBounds();
    const int width = bounds.x2 - bounds.x1;
    const int height = bounds.y2 - bounds.y1;

    float* input = static_cast<float*>(_srcImg->getPixelData());
    float* output = static_cast<float*>(_dstImg->getPixelData());

    RunOpenCLKernel(_pOpenCLCmdQ, width, height, _switch, _scales, input, output);
}


void TwelveWay::multiThreadProcessImages(OfxRectI p_ProcWindow)
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
                  
			      float Ro = srcPix[0] * _scales[2] + _scales[0] * (1.0 - (srcPix[0] * _scales[2]));
                  float RO = Ro >= 0.0 && Ro <= 1.0 ? (_switch[0] != 1.0 ? pow(Ro, 1.0 / _scales[1]) : 1.0 - pow(1.0 - Ro, _scales[1])) : Ro;
                  
                  float Go = srcPix[1] * _scales[2] + _scales[0] * (1.0 - (srcPix[1] * _scales[2]));
                  float GO = Go >= 0.0 && Go <= 1.0 ? (_switch[0] != 1.0 ? pow(Go, 1.0 / _scales[1]) : 1.0 - pow(1.0 - Go, _scales[1])) : Go;
                  
                  float Bo = srcPix[2] * _scales[2] + _scales[0] * (1.0 - (srcPix[2] * _scales[2]));
                  float BO = Bo >= 0.0 && Bo <= 1.0 ? (_switch[0] != 1.0 ? pow(Bo, 1.0 / _scales[1]) : 1.0 - pow(1.0 - Bo, _scales[1])) : Bo;
                  
                  float Rs = (RO - _scales[6]) / (_scales[7] - _scales[6]);
                  float Rss = Rs >= 0.0 && Rs <= 1.0 ? Rs * _scales[5] + _scales[3] * (1.0 - (Rs * _scales[5])) : Rs;
                  float RS = Rss >= 0.0 && Rss <= 1.0 ? (_switch[1] != 1.0 ? pow(Rss, 1.0 / _scales[4]) : 1.0 - pow(1.0 - Rss, _scales[4])) : Rss;
                  float RSS = RS * (_scales[7] - _scales[6]) + _scales[6];
                  
                  float Gs = (GO - _scales[6]) / (_scales[7] - _scales[6]);
                  float Gss = Gs >= 0.0 && Gs <= 1.0 ? Gs * _scales[5] + _scales[3] * (1.0 - (Gs * _scales[5])) : Gs;
                  float GS = Gss >= 0.0 && Gss <= 1.0 ? (_switch[1] != 1.0 ? pow(Gss, 1.0 / _scales[4]) : 1.0 - pow(1.0 - Gss, _scales[4])) : Gss;
                  float GSS = GS * (_scales[7] - _scales[6]) + _scales[6];
                  
                  float Bs = (BO - _scales[6]) / (_scales[7] - _scales[6]);
                  float Bss = Bs >= 0.0 && Bs <= 1.0 ? Bs * _scales[5] + _scales[3] * (1.0 - (Bs * _scales[5])) : Bs;
                  float BS = Bss >= 0.0 && Bss <= 1.0 ? (_switch[1] != 1.0 ? pow(Bss, 1.0 / _scales[4]) : 1.0 - pow(1.0 - Bss, _scales[4])) : Bss;
                  float BSS = BS * (_scales[7] - _scales[6]) + _scales[6];
                
                  float Rm = (RSS - _scales[11]) / (_scales[12] - _scales[11]);
                  float Rmm = Rm >= 0.0 && Rm <= 1.0 ? Rm * _scales[10] + _scales[8] * (1.0 - (Rm * _scales[10])) : Rm;
                  float RM = Rmm >= 0.0 && Rmm <= 1.0 ? (_switch[2] != 1.0 ? pow(Rmm, 1.0 / _scales[9]) : 1.0 - pow(1.0 - Rmm, _scales[9])) : Rmm;
                  float RMM = RM * (_scales[12] - _scales[11]) + _scales[11];
                  
                  float Gm = (GSS - _scales[11]) / (_scales[12] - _scales[11]);
                  float Gmm = Gm >= 0.0 && Gm <= 1.0 ? Gm * _scales[10] + _scales[8] * (1.0 - (Gm * _scales[10])) : Gm;
                  float GM = Gmm >= 0.0 && Gmm <= 1.0 ? (_switch[2] != 1.0 ? pow(Gmm, 1.0 / _scales[9]) : 1.0 - pow(1.0 - Gmm, _scales[9])) : Gmm;
                  float GMM = GM * (_scales[12] - _scales[11]) + _scales[11];
                  
                  float Bm = (BSS - _scales[11]) / (_scales[12] - _scales[11]);
                  float Bmm = Bm >= 0.0 && Bm <= 1.0 ? Bm * _scales[10] + _scales[8] * (1.0 - (Bm * _scales[10])) : Bm;
                  float BM = Bmm >= 0.0 && Bmm <= 1.0 ? (_switch[2] != 1.0 ? pow(Bmm, 1.0 / _scales[9]) : 1.0 - pow(1.0 - Bmm, _scales[9])) : Bmm;
                  float BMM = BM * (_scales[12] - _scales[11]) + _scales[11];
                  
                  float Rh = (RMM - _scales[16]) / (_scales[17] - _scales[16]);
                  float Rhh = Rh >= 0.0 && Rh <= 1.0 ? Rh * _scales[15] + _scales[13] * (1.0 - (Rh * _scales[15])) : Rh;
                  float RH = Rhh >= 0.0 && Rhh <= 1.0 ? (_switch[3] != 1.0 ? pow(Rhh, 1.0 / _scales[14]) : 1.0 - pow(1.0 - Rhh, _scales[14])) : Rhh;
                  float RHH = RH * (_scales[17] - _scales[16]) + _scales[16];
                  
                  float Gh = (GMM - _scales[16]) / (_scales[17] - _scales[16]);
                  float Ghh = Gh >= 0.0 && Gh <= 1.0 ? Gh * _scales[15] + _scales[13] * (1.0 - (Gh * _scales[15])) : Gh;
                  float GH = Ghh >= 0.0 && Ghh <= 1.0 ? (_switch[3] != 1.0 ? pow(Ghh, 1.0 / _scales[14]) : 1.0 - pow(1.0 - Ghh, _scales[14])) : Ghh;
                  float GHH = GH * (_scales[17] - _scales[16]) + _scales[16];
                  
                  float Bh = (BMM - _scales[16]) / (_scales[17] - _scales[16]);
                  float Bhh = Bh >= 0.0 && Bh <= 1.0 ? Bh * _scales[15] + _scales[13] * (1.0 - (Bh * _scales[15])) : Bh;
                  float BH = Bhh >= 0.0 && Bhh <= 1.0 ? (_switch[3] != 1.0 ? pow(Bhh, 1.0 / _scales[14]) : 1.0 - pow(1.0 - Bhh, _scales[14])) : Bhh;
                  float BHH = BH * (_scales[17] - _scales[16]) + _scales[16];
                  
                  dstPix[0] = RHH;
                  dstPix[1] = GHH;
                  dstPix[2] = BHH;
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

void TwelveWay::setSrcImg(OFX::Image* p_SrcImg)
{
    _srcImg = p_SrcImg;
}

void TwelveWay::setScales(float p_SwitchO, float p_SwitchS, float p_SwitchM, float p_SwitchH, float p_ScaleOL, float p_ScaleOG, float p_ScaleOGG, 
    float p_ScaleSL, float p_ScaleSG, float p_ScaleSGG, float p_ScaleSA, float p_ScaleSB, 
    float p_ScaleML, float p_ScaleMG,  float p_ScaleMGG, float p_ScaleMA, float p_ScaleMB, 
    float p_ScaleHL, float p_ScaleHG,  float p_ScaleHGG, float p_ScaleHA, float p_ScaleHB)
{
    _switch[0] = p_SwitchO;
    _switch[1] = p_SwitchS;
    _switch[2] = p_SwitchM;
    _switch[3] = p_SwitchH;
    _scales[0] = p_ScaleOL;
    _scales[1] = p_ScaleOG;
    _scales[2] = p_ScaleOGG;
    _scales[3] = p_ScaleSL;
    _scales[4] = p_ScaleSG;
    _scales[5] = p_ScaleSGG;
    _scales[6] = p_ScaleSA;
    _scales[7] = p_ScaleSB;
    _scales[8] = p_ScaleML;
    _scales[9] = p_ScaleMG;
    _scales[10] = p_ScaleMGG;
    _scales[11] = p_ScaleMA;
    _scales[12] = p_ScaleMB;
    _scales[13] = p_ScaleHL;
    _scales[14] = p_ScaleHG;
    _scales[15] = p_ScaleHGG;
    _scales[16] = p_ScaleHA;
    _scales[17] = p_ScaleHB;
}

////////////////////////////////////////////////////////////////////////////////
/** @brief The plugin that does our work */
class TwelveWayPlugin : public OFX::ImageEffect
{
public:
    explicit TwelveWayPlugin(OfxImageEffectHandle p_Handle);

    /* Override the render */
    virtual void render(const OFX::RenderArguments& p_Args);

    /* Override is identity */
    virtual bool isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime);
    
    /* Override changedParam */
    virtual void changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName);

    /* Set up and run a processor */
    void setupAndProcess(TwelveWay &p_TwelveWay, const OFX::RenderArguments& p_Args);

private:
    // Does not own the following pointers
    OFX::Clip* m_DstClip;
    OFX::Clip* m_SrcClip;

	OFX::BooleanParam* m_SwitchO;
	OFX::BooleanParam* m_SwitchS;
	OFX::BooleanParam* m_SwitchM;
	OFX::BooleanParam* m_SwitchH;
    OFX::DoubleParam* m_ScaleOL;
    OFX::DoubleParam* m_ScaleOG;
    OFX::DoubleParam* m_ScaleOGG;
    OFX::DoubleParam* m_ScaleSL;
    OFX::DoubleParam* m_ScaleSG;
    OFX::DoubleParam* m_ScaleSGG;
    OFX::DoubleParam* m_ScaleSA;
    OFX::DoubleParam* m_ScaleSB;
    OFX::DoubleParam* m_ScaleML;
    OFX::DoubleParam* m_ScaleMG;
    OFX::DoubleParam* m_ScaleMGG;
    OFX::DoubleParam* m_ScaleMA;
    OFX::DoubleParam* m_ScaleMB;
    OFX::DoubleParam* m_ScaleHL;
    OFX::DoubleParam* m_ScaleHG;
    OFX::DoubleParam* m_ScaleHGG;
    OFX::DoubleParam* m_ScaleHA;
    OFX::DoubleParam* m_ScaleHB;
    OFX::StringParam* m_Path;
    OFX::StringParam* m_Name;
    OFX::PushButtonParam* m_Info;
	OFX::PushButtonParam* m_Button1;
	OFX::PushButtonParam* m_Button2;
};

TwelveWayPlugin::TwelveWayPlugin(OfxImageEffectHandle p_Handle)
    : ImageEffect(p_Handle)
{
    m_DstClip = fetchClip(kOfxImageEffectOutputClipName);
    m_SrcClip = fetchClip(kOfxImageEffectSimpleSourceClipName);

	m_SwitchO = fetchBooleanParam("switchO");
	m_SwitchS = fetchBooleanParam("switchS");
	m_SwitchM = fetchBooleanParam("switchM");
	m_SwitchH = fetchBooleanParam("switchH");
    m_ScaleOL = fetchDoubleParam("scaleOL");
    m_ScaleOG = fetchDoubleParam("scaleOG");
    m_ScaleOGG = fetchDoubleParam("scaleOGG");
    m_ScaleSL = fetchDoubleParam("scaleSL");
    m_ScaleSG = fetchDoubleParam("scaleSG");
    m_ScaleSGG = fetchDoubleParam("scaleSGG");
    m_ScaleSA = fetchDoubleParam("scaleSA");
    m_ScaleSB = fetchDoubleParam("scaleSB");
    m_ScaleML = fetchDoubleParam("scaleML");
    m_ScaleMG = fetchDoubleParam("scaleMG");
    m_ScaleMGG = fetchDoubleParam("scaleMGG");
    m_ScaleMA = fetchDoubleParam("scaleMA");
    m_ScaleMB = fetchDoubleParam("scaleMB");
    m_ScaleHL = fetchDoubleParam("scaleHL");
    m_ScaleHG = fetchDoubleParam("scaleHG");
    m_ScaleHGG = fetchDoubleParam("scaleHGG");
    m_ScaleHA = fetchDoubleParam("scaleHA");
    m_ScaleHB = fetchDoubleParam("scaleHB");
    m_Path = fetchStringParam("path");
	m_Name = fetchStringParam("name");
	m_Info = fetchPushButtonParam("info");
	m_Button1 = fetchPushButtonParam("button1");
	m_Button2 = fetchPushButtonParam("button2");
    
}

void TwelveWayPlugin::render(const OFX::RenderArguments& p_Args)
{
    if ((m_DstClip->getPixelDepth() == OFX::eBitDepthFloat) && (m_DstClip->getPixelComponents() == OFX::ePixelComponentRGBA))
    {
        TwelveWay TwelveWay(*this);
        setupAndProcess(TwelveWay, p_Args);
    }
    else
    {
        OFX::throwSuiteStatusException(kOfxStatErrUnsupported);
    }
}

bool TwelveWayPlugin::isIdentity(const OFX::IsIdentityArguments& p_Args, OFX::Clip*& p_IdentityClip, double& p_IdentityTime)
{

    double olScale = m_ScaleOL->getValueAtTime(p_Args.time);
    double ogScale = m_ScaleOG->getValueAtTime(p_Args.time);
    double oggScale = m_ScaleOGG->getValueAtTime(p_Args.time);
    double slScale = m_ScaleSL->getValueAtTime(p_Args.time);
    double sgScale = m_ScaleSG->getValueAtTime(p_Args.time);
    double sggScale = m_ScaleSGG->getValueAtTime(p_Args.time);
    double saScale = m_ScaleSA->getValueAtTime(p_Args.time);
    double sbScale = m_ScaleSB->getValueAtTime(p_Args.time);
    double mlScale = m_ScaleML->getValueAtTime(p_Args.time);
    double mgScale = m_ScaleMG->getValueAtTime(p_Args.time);
    double mggScale = m_ScaleMGG->getValueAtTime(p_Args.time);
    double maScale = m_ScaleMA->getValueAtTime(p_Args.time);
    double mbScale = m_ScaleMB->getValueAtTime(p_Args.time);
    double hlScale = m_ScaleHL->getValueAtTime(p_Args.time);
    double hgScale = m_ScaleHG->getValueAtTime(p_Args.time);
    double hggScale = m_ScaleHGG->getValueAtTime(p_Args.time);
    double haScale = m_ScaleHA->getValueAtTime(p_Args.time);
    double hbScale = m_ScaleHB->getValueAtTime(p_Args.time);
    

    if ((olScale == 0.0) && (ogScale == 1.0) && (oggScale == 1.0) && (slScale == 0.0) && (sgScale == 1.0) && (sggScale == 1.0)
    	 && (mlScale == 0.0) && (mgScale == 1.0) && (mggScale == 1.0) && (hlScale == 0.0) && (hgScale == 1.0) && (hggScale == 1.0))
    {
        p_IdentityClip = m_SrcClip;
        p_IdentityTime = p_Args.time;
        return true;
    }

    return false;
}

void TwelveWayPlugin::changedParam(const OFX::InstanceChangedArgs& p_Args, const std::string& p_ParamName)
{
 
 	if(p_ParamName == "info")
    {
	
	sendMessage(OFX::Message::eMessageMessage, "", string(kPluginDescription));
	
	}
	
	if(p_ParamName == "button1")
    {
    
    bool oSwitch = m_SwitchO->getValueAtTime(p_Args.time);
    bool sSwitch = m_SwitchS->getValueAtTime(p_Args.time);
    bool mSwitch = m_SwitchM->getValueAtTime(p_Args.time);
    bool hSwitch = m_SwitchH->getValueAtTime(p_Args.time);

	int overall = oSwitch ? 1 : 0;
	int shadow = sSwitch ? 1 : 0;
	int mid = mSwitch ? 1 : 0;
	int highlight = hSwitch ? 1 : 0;

    float lift = m_ScaleOL->getValueAtTime(p_Args.time);
    float gamma = m_ScaleOG->getValueAtTime(p_Args.time);
    float gain = m_ScaleOGG->getValueAtTime(p_Args.time);
    float liftS = m_ScaleSL->getValueAtTime(p_Args.time);
    float gammaS = m_ScaleSG->getValueAtTime(p_Args.time);
    float gainS = m_ScaleSGG->getValueAtTime(p_Args.time);
    float shadowA = m_ScaleSA->getValueAtTime(p_Args.time);
    float shadowB = m_ScaleSB->getValueAtTime(p_Args.time);
    float liftM = m_ScaleML->getValueAtTime(p_Args.time);
    float gammaM = m_ScaleMG->getValueAtTime(p_Args.time);
    float gainM = m_ScaleMGG->getValueAtTime(p_Args.time);
    float midA = m_ScaleMA->getValueAtTime(p_Args.time);
    float midB = m_ScaleMB->getValueAtTime(p_Args.time);
    float liftH = m_ScaleHL->getValueAtTime(p_Args.time);
    float gammaH = m_ScaleHG->getValueAtTime(p_Args.time);
    float gainH = m_ScaleHGG->getValueAtTime(p_Args.time);
    float highA = m_ScaleHA->getValueAtTime(p_Args.time);
    float highB = m_ScaleHB->getValueAtTime(p_Args.time);
    
    string PATH;
	m_Path->getValue(PATH);
	
	string NAME;
	m_Name->getValue(NAME);
	
	OFX::Message::MessageReplyEnum reply = sendMessage(OFX::Message::eMessageQuestion, "", "Save " + NAME + ".dctl to " + PATH + "?");
	if (reply == OFX::Message::eMessageReplyYes) {
	
	FILE * pFile;
	
	pFile = fopen ((PATH + "/" + NAME + ".dctl").c_str(), "w");
	if (pFile != NULL) {
    	
	fprintf (pFile, "// TwelveWayPlugin DCTL export\n" \
	"\n" \
	"__DEVICE__ float3 transform(int p_Width, int p_Height, int p_X, int p_Y, float p_R, float p_G, float p_B)\n" \
	"{\n" \
	"    \n" \
	"    // switches for overall, shadows, midtones, highlights\n" \
	"	int overall = %d;\n" \
	"	int shadow = %d;\n" \
	"	int mid = %d;\n" \
	"	int highlight = %d;\n" \
	"	bool p_SwitchOverall = switchO == 1;\n" \
	"	bool p_SwitchShadows = switchS == 1;\n" \
	"	bool p_SwitchMidtones = switchM == 1;\n" \
	"	bool p_SwitchHighlights = switchH == 1;\n" \
	"	\n" \
	"	// Parameter values lift, gamma, gain\n" \
	"	float lift = %ff;\n" \
	"	float gamma = %ff;\n" \
	"	float gain = %ff;\n" \
	"	\n" \
	"	float liftS = %ff;\n" \
	"	float gammaS = %ff;\n" \
	"	float gainS = %ff;\n" \
	"	float shadowA = %ff;\n" \
	"	float shadowB = %ff;\n" \
	"	\n" \
	"	float liftM = %ff;\n" \
	"	float gammaM = %ff;\n" \
	"	float gainM = %ff;\n" \
	"	float midA = %ff;\n" \
	"	float midB = %ff;\n" \
	"	\n" \
	"	float liftH = %ff;\n" \
	"	float gammaH = %ff;\n" \
	"	float gainH = %ff;\n" \
	"	float highA = %ff;\n" \
	"	float highB = %ff;	\n" \
	"   \n" \
	"	float red = (p_R * gain) + (lift * (1.0f - (p_R * gain)));\n" \
	"	float green = (p_G * gain) + (lift * (1.0f - (p_G * gain)));\n" \
	"	float blue = (p_B * gain) + (lift * (1.0f - (p_B * gain)));\n" \
	"	\n" \
	"	const float RO = !p_SwitchOverall ? _powf(red, 1.0f/gamma) : 1.0f - _powf(1.0f - red, gamma);\n" \
	"	const float GO = !p_SwitchOverall ? _powf(green, 1.0f/gamma) : 1.0f - _powf(1.0f - green, gamma);\n" \
	"	const float BO = !p_SwitchOverall ? _powf(blue, 1.0f/gamma) : 1.0f - _powf(1.0f - blue, gamma);\n" \
	"  \n" \
	"	float Rs = (((RO - shadowA) / (shadowB - shadowA)) * gainS) + (liftS * (1.0f - (RO - shadowA) / (shadowB - shadowA)));\n" \
	"	float RS = RO >= shadowA && RO <= shadowB ? ((!p_SwitchShadows ? _powf(Rs, 1.0f / gammaS) : 1.0f - _powf(1.0f - Rs, gammaS)) * (shadowB - shadowA)) + shadowA : RO;\n" \
	"	\n" \
	"	float Gs = (((GO - shadowA) / (shadowB - shadowA)) * gainS) + (liftS * (1.0f - (GO - shadowA) / (shadowB - shadowA)));\n" \
	"	float GS = GO >= shadowA && GO <= shadowB ? ((!p_SwitchShadows ? _powf(Gs, 1.0f / gammaS) : 1.0f - _powf(1.0f - Gs, gammaS)) * (shadowB - shadowA)) + shadowA : GO;\n" \
	"	\n" \
	"	float Bs = (((BO - shadowA) / (shadowB - shadowA)) * gainS) + (liftS * (1.0f - (BO - shadowA) / (shadowB - shadowA)));\n" \
	"	float BS = BO >= shadowA && BO <= shadowB ? ((!p_SwitchShadows ? _powf(Bs, 1.0f / gammaS) : 1.0f - _powf(1.0f - Bs, gammaS)) * (shadowB - shadowA)) + shadowA : BO;\n" \
	"	\n" \
	"	float Rm = (((RS - midA) / (midB - midA)) * gainM) + (LiftM * (1.0f - (RS - midA) / (midB - midA)));\n" \
	"	float RM = RS >= midA && RS <= midB ? ((!p_SwitchMidtones ? _powf(Rm, 1.0f / LiftM) : 1.0f - _powf(1.0f - Rm, LiftM)) * (midB - midA)) + midA : RS;\n" \
	"	\n" \
	"	float Gm = (((GS - midA) / (midB - midA)) * gainM) + (LiftM * (1.0f - (GS - midA) / (midB - midA)));\n" \
	"	float GM = GS >= midA && GS <= midB ? ((!p_SwitchMidtones ? _powf(Gm, 1.0f / LiftM) : 1.0f - _powf(1.0f - Gm, LiftM)) * (midB - midA)) + midA : GS;\n" \
	"	\n" \
	"	float Bm = (((BS - midA) / (midB - midA)) * gainM) + (LiftM * (1.0f - (BS - midA) / (midB - midA)));\n" \
	"	float BM = BS >= midA && BS <= midB ? ((!p_SwitchMidtones ? _powf(Bm, 1.0f / LiftM) : 1.0f - _powf(1.0f - Bm, LiftM)) * (midB - midA)) + midA : BS;\n" \
	"	\n" \
	"	float Rh = (((RM - highA) / (highB - highA)) * gainH) + (liftH * (1.0f - (RM - highA) / (highB - highA)));\n" \
	"	float RH = RM >= highA && RM <= highB ? ((!p_SwitchHighlights ? _powf(Rh, 1.0f / gammaH) : 1.0f - _powf(1.0f - Rh, gammaH)) * (highB - highA)) + highA : RM;\n" \
	"	\n" \
	"	float Gh = (((GM - highA) / (highB - highA)) * gainH) + (liftH * (1.0f - (GM - highA) / (highB - highA)));\n" \
	"	float GH = GM >= highA && GM <= highB ? ((!p_SwitchHighlights ? _powf(Gh, 1.0f / gammaH) : 1.0f - _powf(1.0f - Gh, gammaH)) * (highB - highA)) + highA : GM;\n" \
	"	\n" \
	"	float Bh = (((BM - highA) / (highB - highA)) * gainH) + (liftH * (1.0f - (BM - highA) / (highB - highA)));\n" \
	"	float BH = BM >= highA && BM <= highB ? ((!p_SwitchHighlights ? _powf(Bh, 1.0f / gammaH) : 1.0f - _powf(1.0f - Bh, gammaH)) * (highB - highA)) + highA : BM;\n" \
	"	\n" \
	"    \n" \
	"    const float r = RH;\n" \
	"    const float g = GH;\n" \
	"    const float b = BH;\n" \
	"\n" \
	"    return make_float3(r, g, b);\n" \
	"}\n", overall, shadow, mid, highlight, lift, gamma, gain, liftS, gammaS, gainS, shadowA, shadowB, liftM, gammaM, gainM, midA, midB, liftH, gammaH, gainH, highA, highB);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".dctl to " + PATH  + ". Check Permissions."));
	}	
	}
	}
	
	if(p_ParamName == "button2")
    {
	
	bool oSwitch = m_SwitchO->getValueAtTime(p_Args.time);
    bool sSwitch = m_SwitchS->getValueAtTime(p_Args.time);
    bool mSwitch = m_SwitchM->getValueAtTime(p_Args.time);
    bool hSwitch = m_SwitchH->getValueAtTime(p_Args.time);

	int overall = oSwitch ? 1 : 0;
	int shadow = sSwitch ? 1 : 0;
	int mid = mSwitch ? 1 : 0;
	int highlight = hSwitch ? 1 : 0;

    float lift = m_ScaleOL->getValueAtTime(p_Args.time);
    float gamma = m_ScaleOG->getValueAtTime(p_Args.time);
    float gain = m_ScaleOGG->getValueAtTime(p_Args.time);
    float liftS = m_ScaleSL->getValueAtTime(p_Args.time);
    float gammaS = m_ScaleSG->getValueAtTime(p_Args.time);
    float gainS = m_ScaleSGG->getValueAtTime(p_Args.time);
    float shadowA = m_ScaleSA->getValueAtTime(p_Args.time);
    float shadowB = m_ScaleSB->getValueAtTime(p_Args.time);
    float liftM = m_ScaleML->getValueAtTime(p_Args.time);
    float gammaM = m_ScaleMG->getValueAtTime(p_Args.time);
    float gainM = m_ScaleMGG->getValueAtTime(p_Args.time);
    float midA = m_ScaleMA->getValueAtTime(p_Args.time);
    float midB = m_ScaleMB->getValueAtTime(p_Args.time);
    float liftH = m_ScaleHL->getValueAtTime(p_Args.time);
    float gammaH = m_ScaleHG->getValueAtTime(p_Args.time);
    float gainH = m_ScaleHGG->getValueAtTime(p_Args.time);
    float highA = m_ScaleHA->getValueAtTime(p_Args.time);
    float highB = m_ScaleHB->getValueAtTime(p_Args.time);
    
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
	" name TwelveWay\n" \
	" xpos -82\n" \
	" ypos 102\n" \
	"}\n" \
	" Input {\n" \
	"  inputs 0\n" \
	"  name Input1\n" \
	"  xpos -171\n" \
	"  ypos -175\n" \
	" }\n" \
	" Expression {\n" \
	"  temp_name0 lift\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 gain\n" \
	"  temp_expr1 %f\n" \
	"  expr0 \"r * gain + lift * (1.0 - (r * gain))\"\n" \
	"  expr1 \"g * gain + lift * (1.0 - (g * gain))\"\n" \
	"  expr2 \"b * gain + lift * (1.0 - (b * gain))\"\n" \
	"  name overall_lift_gain\n" \
	"  xpos -171\n" \
	"  ypos -132\n" \
	" }\n" \
	"set N26456900 [stack 0]\n" \
	" Expression {\n" \
	"  temp_name0 gamma\n" \
	"  temp_expr0 %f\n" \
	"  expr0 \"1.0 - pow(1.0 - r, gamma)\"\n" \
	"  expr1 \"1.0 - pow(1.0 - b, gamma)\"\n" \
	"  expr2 \"1.0 - pow(1.0 - b, gamma)\"\n" \
	"  name overall_gamma_upper\n" \
	"  xpos -231\n" \
	"  ypos -99\n" \
	" }\n" \
	"push $N26456900\n" \
	" Expression {\n" \
	"  temp_name0 gamma\n" \
	"  temp_expr0 %f\n" \
	"  expr0 \"pow(r, 1.0 / gamma)\"\n" \
	"  expr1 \"pow(g, 1.0 / gamma)\"\n" \
	"  expr2 \"pow(b, 1.0 / gamma)\"\n" \
	"  name overall_gamma_lower\n" \
	"  xpos -117\n" \
	"  ypos -94\n" \
	" }\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name overall_upper\n" \
	"  xpos -171\n" \
	"  ypos -53\n" \
	" }\n" \
	"set N26483490 [stack 0]\n" \
	"push $N26483490\n" \
	" Expression {\n" \
	"  temp_name0 liftS\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 gainS\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 shadowA\n" \
	"  temp_expr2 %f\n" \
	"  temp_name3 shadowB\n" \
	"  temp_expr3 %f\n" \
	"  expr0 \"((r - shadowA) / (shadowB - shadowA)) * gainS + liftS * (1.0 - (r - shadowA) / (shadowB - shadowA))\"\n" \
	"  expr1 \"((g - shadowA) / (shadowB - shadowA)) * gainS + liftS * (1.0 - (g - shadowA) / (shadowB - shadowA))\"\n" \
	"  expr2 \"((b - shadowA) / (shadowB - shadowA)) * gainS + liftS * (1.0 - (b - shadowA) / (shadowB - shadowA))\"\n" \
	"  name shadow_lift_gain\n" \
	"  xpos -122\n" \
	"  ypos -19\n" \
	" }\n" \
	"set N26693880 [stack 0]\n" \
	" MergeExpression {\n" \
	"  inputs 2\n" \
	"  temp_name0 gammaS\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 shadowA\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 shadowB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"Ar >= shadowA && Ar <= shadowB ? (1.0 - pow(1.0 - Br, gammaS)  * (shadowB - shadowA)) + shadowA : Ar\"\n" \
	"  expr1 \"Ag >= shadowA && Ag <= shadowB ? (1.0 - pow(1.0 - Bg, gammaS)  * (shadowB - shadowA)) + shadowA : Ag\"\n" \
	"  expr2 \"Ab >= shadowA && Ab <= shadowB ? (1.0 - pow(1.0 - Bb, gammaS)  * (shadowB - shadowA)) + shadowA : Ab\"\n" \
	"  name shadow_gamma_upper\n" \
	"  xpos -122\n" \
	"  ypos 23\n" \
	" }\n" \
	"push $N26483490\n" \
	"push $N26693880\n" \
	" MergeExpression {\n" \
	"  inputs 2\n" \
	"  temp_name0 gammaS\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 shadowA\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 shadowB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"Ar >= shadowA && Ar <= shadowB ? (pow(Br, 1.0 / gammaS)  * (shadowB - shadowA)) + shadowA : Ar\"\n" \
	"  expr1 \"Ag >= shadowA && Ag <= shadowB ? (pow(Bg, 1.0 / gammaS)  * (shadowB - shadowA)) + shadowA : Ag\"\n" \
	"  expr2 \"Ab >= shadowA && Ab <= shadowB ? (pow(Bb, 1.0 / gammaS)  * (shadowB - shadowA)) + shadowA : Ab\"\n" \
	"  name shadow_gamma_lower\n" \
	"  xpos -225\n" \
	"  ypos 25\n" \
	" }\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name shadow_upper\n" \
	"  xpos -170\n" \
	"  ypos 61\n" \
	" }\n" \
	"set N25a93740 [stack 0]\n" \
	"push $N25a93740\n" \
	" Expression {\n" \
	"  temp_name0 liftM\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 gainM\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 midA\n" \
	"  temp_expr2 %f\n" \
	"  temp_name3 midB\n" \
	"  temp_expr3 %f\n" \
	"  expr0 \"((r - midA) / (midB - midA)) * gainM + liftM * (1.0 - (r - midA) / (midB - midA))\"\n" \
	"  expr1 \"((g - midA) / (midB - midA)) * gainM + liftM * (1.0 - (g - midA) / (midB - midA))\"\n" \
	"  expr2 \"((b - midA) / (midB - midA)) * gainM + liftM * (1.0 - (b - midA) / (midB - midA))\"\n" \
	"  name mid_lift_gain\n" \
	"  xpos -129\n" \
	"  ypos 99\n" \
	" }\n" \
	"set N257d59a0 [stack 0]\n" \
	" MergeExpression {\n" \
	"  inputs 2\n" \
	"  temp_name0 gammaM\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 midA\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 midB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"Ar >= midA && Ar <= midB ? (1.0 - pow(1.0 - Br, gammaM)  * (midB - midA)) + midA : Ar\"\n" \
	"  expr1 \"Ag >= midA && Ag <= midB ? (1.0 - pow(1.0 - Bg, gammaM)  * (midB - midA)) + midA : Ag\"\n" \
	"  expr2 \"Ab >= midA && Ab <= midB ? (1.0 - pow(1.0 - Bb, gammaM)  * (midB - midA)) + midA : Ab\"\n" \
	"  name mid_gamma_upper\n" \
	"  xpos -115\n" \
	"  ypos 137\n" \
	" }\n" \
	"push $N25a93740\n" \
	"push $N257d59a0\n" \
	" MergeExpression {\n" \
	"  inputs 2\n" \
	"  temp_name0 gammaM\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 midA\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 midB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"Ar >= midA && Ar <= midB ? (pow(Br, 1.0 / gammaM)  * (midB - midA)) + midA : Ar\"\n" \
	"  expr1 \"Ag >= midA && Ag <= midB ? (pow(Bg, 1.0 / gammaM)  * (midB - midA)) + midA : Ag\"\n" \
	"  expr2 \"Ab >= midA && Ab <= midB ? (pow(Bb, 1.0 / gammaM)  * (midB - midA)) + midA : Ab\"\n" \
	"  name mid_gamma_lower\n" \
	"  xpos -229\n" \
	"  ypos 136\n" \
	" }\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name mid_upper\n" \
	"  xpos -159\n" \
	"  ypos 178\n" \
	" }\n" \
	"set N257280c0 [stack 0]\n" \
	"push $N257280c0\n" \
	" Expression {\n" \
	"  temp_name0 liftH\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 gainH\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 highA\n" \
	"  temp_expr2 %f\n" \
	"  temp_name3 highB\n" \
	"  temp_expr3 %f\n" \
	"  expr0 \"((r - highA) / (highB - highA)) * gainH + liftH * (1.0 - (r - highA) / (highB - highA))\"\n" \
	"  expr1 \"((g - highA) / (highB - highA)) * gainH + liftH * (1.0 - (g - highA) / (highB - highA))\"\n" \
	"  expr2 \"((b - highA) / (highB - highA)) * gainH + liftH * (1.0 - (b - highA) / (highB - highA))\"\n" \
	"  name high_lift_gain\n" \
	"  xpos -118\n" \
	"  ypos 211\n" \
	" }\n" \
	"set N25a20b50 [stack 0]\n" \
	" MergeExpression {\n" \
	"  inputs 2\n" \
	"  temp_name0 gammaH\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 highA\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 highB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"Ar >= highA && Ar <= highB ? (1.0 - pow(1.0 - Br, gammaH)  * (highB - highA)) + highA : Ar\"\n" \
	"  expr1 \"Ag >= highA && Ag <= highB ? (1.0 - pow(1.0 - Bg, gammaH)  * (highB - highA)) + highA : Ag\"\n" \
	"  expr2 \"Ab >= highA && Ab <= highB ? (1.0 - pow(1.0 - Bb, gammaH)  * (highB - highA)) + highA : Ab\"\n" \
	"  name high_gamma_upper\n" \
	"  xpos -103\n" \
	"  ypos 252\n" \
	" }\n" \
	"push $N257280c0\n" \
	"push $N25a20b50\n" \
	" MergeExpression {\n" \
	"  inputs 2\n" \
	"  temp_name0 gammaH\n" \
	"  temp_expr0 %f\n" \
	"  temp_name1 highA\n" \
	"  temp_expr1 %f\n" \
	"  temp_name2 highB\n" \
	"  temp_expr2 %f\n" \
	"  expr0 \"Ar >= highA && Ar <= highB ? (pow(Br, 1.0 / gammaH)  * (highB - highA)) + highA : Ar\"\n" \
	"  expr1 \"Ag >= highA && Ag <= highB ? (pow(Bg, 1.0 / gammaH)  * (highB - highA)) + highA : Ag\"\n" \
	"  expr2 \"Ab >= highA && Ab <= highB ? (pow(Bg, 1.0 / gammaH)  * (highB - highA)) + highA : Ag\"\n" \
	"  name high_gamma_lower\n" \
	"  xpos -215\n" \
	"  ypos 254\n" \
	" }\n" \
	" Switch {\n" \
	"  inputs 2\n" \
	"  which %d\n" \
	"  name high_upper\n" \
	"  xpos -157\n" \
	"  ypos 301\n" \
	" }\n" \
	" Output {\n" \
	"  name Output1\n" \
	"  xpos -157\n" \
	"  ypos 342\n" \
	" }\n" \
	"end_group\n", lift, gain, gamma, gamma, overall, liftS, gainS, shadowA, shadowB, gammaS, shadowA, shadowB, 
	gammaS, shadowA, shadowB, shadow, liftM, gainM, midA, midB, gammaM, midA, midB, gammaM, midA, midB, mid, 
	liftH, gainH, highA, highB, gammaH, highA, highB, gammaH, highA, highB, highlight);
	fclose (pFile);
	} else {
     sendMessage(OFX::Message::eMessageError, "", string("Error: Cannot save " + NAME + ".nk to " + PATH  + ". Check Permissions."));
	}	
	}
	}
}


void TwelveWayPlugin::setupAndProcess(TwelveWay& p_TwelveWay, const OFX::RenderArguments& p_Args)
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

	bool oSwitch = m_SwitchO->getValueAtTime(p_Args.time);
    bool sSwitch = m_SwitchS->getValueAtTime(p_Args.time);
    bool mSwitch = m_SwitchM->getValueAtTime(p_Args.time);
    bool hSwitch = m_SwitchH->getValueAtTime(p_Args.time);

	float oSwitchF = oSwitch ? 1.0f : 0.0f;
	float sSwitchF = sSwitch ? 1.0f : 0.0f;
	float mSwitchF = mSwitch ? 1.0f : 0.0f;
	float hSwitchF = hSwitch ? 1.0f : 0.0f;

    double olScale = m_ScaleOL->getValueAtTime(p_Args.time);
    double ogScale = m_ScaleOG->getValueAtTime(p_Args.time);
    double oggScale = m_ScaleOGG->getValueAtTime(p_Args.time);
    double slScale = m_ScaleSL->getValueAtTime(p_Args.time);
    double sgScale = m_ScaleSG->getValueAtTime(p_Args.time);
    double sggScale = m_ScaleSGG->getValueAtTime(p_Args.time);
    double saScale = m_ScaleSA->getValueAtTime(p_Args.time);
    double sbScale = m_ScaleSB->getValueAtTime(p_Args.time);
    double mlScale = m_ScaleML->getValueAtTime(p_Args.time);
    double mgScale = m_ScaleMG->getValueAtTime(p_Args.time);
    double mggScale = m_ScaleMGG->getValueAtTime(p_Args.time);
    double maScale = m_ScaleMA->getValueAtTime(p_Args.time);
    double mbScale = m_ScaleMB->getValueAtTime(p_Args.time);
    double hlScale = m_ScaleHL->getValueAtTime(p_Args.time);
    double hgScale = m_ScaleHG->getValueAtTime(p_Args.time);
    double hggScale = m_ScaleHGG->getValueAtTime(p_Args.time);
    double haScale = m_ScaleHA->getValueAtTime(p_Args.time);
    double hbScale = m_ScaleHB->getValueAtTime(p_Args.time);

    // Set the images
    p_TwelveWay.setDstImg(dst.get());
    p_TwelveWay.setSrcImg(src.get());

    // Setup OpenCL and CUDA Render arguments
    p_TwelveWay.setGPURenderArgs(p_Args);

    // Set the render window
    p_TwelveWay.setRenderWindow(p_Args.renderWindow);

    // Set the scales
    p_TwelveWay.setScales(oSwitchF, sSwitchF, mSwitchF, hSwitchF, olScale, ogScale, oggScale, 
    slScale, sgScale,sggScale, saScale, sbScale, mlScale, mgScale, mggScale, maScale, mbScale, 
    hlScale, hgScale, hggScale, haScale, hbScale);     

    // Call the base class process member, this will call the derived templated process code
    p_TwelveWay.process();
}

////////////////////////////////////////////////////////////////////////////////

using namespace OFX;

TwelveWayPluginFactory::TwelveWayPluginFactory()
    : OFX::PluginFactoryHelper<TwelveWayPluginFactory>(kPluginIdentifier, kPluginVersionMajor, kPluginVersionMinor)
{
}

void TwelveWayPluginFactory::describe(OFX::ImageEffectDescriptor& p_Desc)
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
	param->setLabel(p_Label);
    param->setScriptName(p_Name);
    param->setHint(p_Hint);
    
    if (p_Parent)
    {
        param->setParent(*p_Parent);
    }

    return param;
}


void TwelveWayPluginFactory::describeInContext(OFX::ImageEffectDescriptor& p_Desc, OFX::ContextEnum /*p_Context*/)
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
 
{    
    GroupParamDescriptor* overall = p_Desc.defineGroupParam("Overall");
    overall->setOpen(true);
    overall->setHint("Overall LGG");
      if (page) {
            page->addChild(*overall);
            }
    
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scaleOL", "Lift", "L from the LGG", overall);
    param->setDefault(0.0);
    param->setRange(-1.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-1.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleOG", "Gamma", "G from the LGG", overall);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("switchO");
    boolParam->setDefault(false);
    boolParam->setHint("higer instead of lower curve bias");
    boolParam->setLabel("Upper Gamma Bias");
    boolParam->setParent(*overall);
    page->addChild(*boolParam);
    
    param = defineScaleParam(p_Desc, "scaleOGG", "Gain", "Double G from the LGG", overall);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
}    
    
{
    GroupParamDescriptor* shadow = p_Desc.defineGroupParam("Shadows");
    shadow->setOpen(false);
    shadow->setHint("Shadows LGG");
      if (page) {
            page->addChild(*shadow);
            }
    
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scaleSL", "Lift", "L from the LGG", shadow);
    param->setDefault(0.0);
    param->setRange(-1.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-1.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleSG", "Gamma", "G from the LGG", shadow);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("switchS");
    boolParam->setDefault(false);
    boolParam->setHint("higer instead of lower curve bias");
    boolParam->setLabel("Upper Gamma Bias");
    boolParam->setParent(*shadow);
    page->addChild(*boolParam);
    
    param = defineScaleParam(p_Desc, "scaleSGG", "Gain", "Double G from the LGG", shadow);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleSA", "Region start", "adjust lower point A", shadow);
    param->setDefault(0.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleSB", "Region end", "adjust upper point B", shadow);
    param->setDefault(0.333);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
}    
 
{    
    GroupParamDescriptor* midtone = p_Desc.defineGroupParam("Midtones");
    midtone->setOpen(false);
    midtone->setHint("Midtones LGG");
      if (page) {
            page->addChild(*midtone);
            }
    
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scaleML", "Lift", "L from the LGG", midtone);
    param->setDefault(0.0);
    param->setRange(-1.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-1.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleMG", "Gamma", "G from the LGG", midtone);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("switchM");
    boolParam->setDefault(false);
    boolParam->setHint("higer instead of lower curve bias");
    boolParam->setLabel("Upper Gamma Bias");
    boolParam->setParent(*midtone);
    page->addChild(*boolParam);
    
    param = defineScaleParam(p_Desc, "scaleMGG", "Gain", "Double G from the LGG", midtone);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleMA", "Region start", "adjust lower point A", midtone);
    param->setDefault(0.333);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleMB", "Region end", "adjust upper point B", midtone);
    param->setDefault(0.667);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
}

{    
    
    GroupParamDescriptor* highlight = p_Desc.defineGroupParam("Highlights");
    highlight->setOpen(false);
    highlight->setHint("Hightlights LGG");
      if (page) {
            page->addChild(*highlight);
            }
    
    DoubleParamDescriptor* param = defineScaleParam(p_Desc, "scaleHL", "Lift", "L from the LGG", highlight);
    param->setDefault(0.0);
    param->setRange(-1.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(-1.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleHG", "Gamma", "G from the LGG", highlight);
    param->setDefault(1.0);
    param->setRange(0.0, 5.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 5.0);
    page->addChild(*param);
    
    BooleanParamDescriptor* boolParam = p_Desc.defineBooleanParam("switchH");
    boolParam->setDefault(false);
    boolParam->setHint("higer instead of lower curve bias");
    boolParam->setLabel("Upper Gamma Bias");
    boolParam->setParent(*highlight);
    page->addChild(*boolParam);
   
    param = defineScaleParam(p_Desc, "scaleHGG", "Gain", "Double G from the LGG", highlight);
    param->setDefault(1.0);
    param->setRange(0.0, 2.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 2.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleHA", "Region start", "adjust lower point A", highlight);
    param->setDefault(0.667);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
    
    param = defineScaleParam(p_Desc, "scaleHB", "Region end", "adjust upper point B", highlight);
    param->setDefault(1.0);
    param->setRange(0.0, 1.0);
    param->setIncrement(0.001);
    param->setDisplayRange(0.0, 1.0);
    page->addChild(*param);
}    
    
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
	param->setDefault("TwelveWay");
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

ImageEffect* TwelveWayPluginFactory::createInstance(OfxImageEffectHandle p_Handle, ContextEnum /*p_Context*/)
{
    return new TwelveWayPlugin(p_Handle);
}

void OFX::Plugin::getPluginIDs(PluginFactoryArray& p_FactoryArray)
{
    static TwelveWayPluginFactory TwelveWayPlugin;
    p_FactoryArray.push_back(&TwelveWayPlugin);
}
