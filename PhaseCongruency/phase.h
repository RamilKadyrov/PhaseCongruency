#pragma once
//#include "options.h"
namespace cv
{
	class Mat;
	class _InputArray;
	class _OutputArray;
//  typedef const _InputArray& InputArray;
//	typedef const _OutputArray& OutputArray;
}

#define CELL_SIZE 64

//void phase_congruency(cv::InputArray _src, cv::OutputArray _dst);
#define norient 8
#define nscale 4
class PhaseCongruency
{

public:
	PhaseCongruency(cv::Size _img_size);
	~PhaseCongruency() {}
	void calc(cv::InputArray _src, cv::OutputArray _edges, cv::OutputArray _corners);
private:
    const double sigma = -1.0 / (2.0 * log(0.65) * log(0.65));

    const float mult = 2.0f;
    const float minwavelength = 1.5f;
    const double epsilon = 0.0002; //0.0001
    const double cutOff = 0.4; //0.4;
    const double g = 10.0;
    const double k = 10.0;
    const int suppres_size = 10;
    const double edge_limit = 0.7;
    const int cell_size = CELL_SIZE;
    const double thin_radius = 1.5;

    cv::Mat	filter[nscale][norient];
    cv::Mat mask[norient];
    cv::Mat pc[norient];

    cv::Mat dft_A, tmp;
    cv::Mat complex[2];


    cv::Mat eo[nscale][norient];


    cv::Mat sumAn;
    cv::Mat sumRe;
    cv::Mat sumIm;
    cv::Mat maxAn;
    cv::Mat energy;
    cv::Mat tmp1;
    cv::Mat tmp2;
    cv::Mat xEnergy;
    cv::Mat tmp3;
    cv::Mat covx2;
    cv::Mat covy2;
    cv::Mat covxy;
    cv::Mat minMoment, maxMoment;
    cv::Mat orientation;
    cv::Mat result;
    cv::Mat  sin2theta;
    cv::Mat  cos2theta;
    cv::Vec<double, 180>  xoff, yoff, hfrac, vfrac;


    void line(cv::OutputArray _dst, float angle);
    void suppres(cv::OutputArray _dst);
};