

//#include "opencv2/core/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <iostream>

#include "phase.h"

#define _USE_MATH_DEFINES
#include <math.h>

//#include "options.h"

#define CV_MINMAX       32

using namespace cv;

// Rearrange the quadrants of Fourier image so that the origin is at
// the image center
// src & dst arrays of equal size & type

void shiftDFT(InputArray _src, OutputArray _dst)
{
    Mat tmp;
       
    Mat src = _src.getMat();

    _dst.create(src.size(), src.type());

    auto dst = _dst.getMat();

    Size size = src.size();

    const int cx = size.width / 2;
    const int cy = size.height / 2; // image center
    
    Mat s1 = src(Rect(0, 0, cx, cy));
    Mat s2 = src(Rect(cx, 0, cx, cy));
    Mat s3 = src(Rect(cx, cy, cx, cy));
    Mat s4 = src(Rect(0, cy, cx, cy));

    Mat d1 = dst(Rect(0, 0, cx, cy));
    Mat d2 = dst(Rect(cx, 0, cx, cy));
    Mat d3 = dst(Rect(cx, cy, cx, cy));
    Mat d4 = dst(Rect(0, cy, cx, cy));

    s3.copyTo(tmp);
    s1.copyTo(d3);
    tmp.copyTo(d1);

    s4.copyTo(tmp);
    s2.copyTo(d4);
    tmp.copyTo(d2);
}

#define MAT_TYPE CV_64FC1
#define MAT_TYPE_CNV CV_64F

// Making a filter
// src & dst arrays of equal size & type
PhaseCongruency::PhaseCongruency(Size _img_size)
{
    const auto dft_M = getOptimalDFTSize(_img_size.height);
    const auto dft_N = getOptimalDFTSize(_img_size.width);

    Mat radius = Mat::zeros(dft_M, dft_N, MAT_TYPE);
    Mat matAr[2];
    matAr[0] = Mat::zeros(dft_M, dft_N, MAT_TYPE);
    matAr[1] = Mat::zeros(dft_M, dft_N, MAT_TYPE);
    Mat lp = Mat::zeros(dft_M, dft_N, MAT_TYPE);
    Mat angular = Mat::zeros(dft_M, dft_N, MAT_TYPE);
    Mat gabor[nscale];

    //Matrix values contain *normalised* radius 
    // values ranging from 0 at the centre to 
    // 0.5 at the boundary.
    int r;
    const int dft_M_2 = dft_M / 2;
    const int dft_N_2 = dft_N / 2;
    if (dft_M > dft_N) r = dft_N_2;
    else r = dft_M_2;
    const double dr = 1.0 / static_cast<double>(r);
    for (int row = dft_M_2 - r; row < dft_M_2 + r; row++)
    {
        auto radius_row = radius.ptr<double>(row);
        for (int col = dft_N_2 - r; col < dft_N_2 + r; col++)
        {
            int m = (row - dft_M_2);
            int n = (col - dft_N_2);
            radius_row[col] = sqrt(static_cast<double>(m * m + n * n)) * dr;
        }
    }
    lp = radius * 2.5; 
    pow(lp, 20.0, lp);
    lp += Scalar::all(1.0); 
    radius.at<double>(dft_M_2, dft_N_2) = 1.0;
    // The following implements the log-gabor transfer function.
    double mt = 1.0f;
    for (int scale = 0; scale < nscale; scale++)
    {
        const double wavelength = minwavelength * mt;
        gabor[scale] = radius * wavelength;
        log(gabor[scale], gabor[scale]);
        pow(gabor[scale], 2.0, gabor[scale]);
        gabor[scale] *= sigma; 
        exp(gabor[scale], gabor[scale]);
        gabor[scale].at<double>(dft_M_2, dft_N_2) = 0.0; 
        divide(gabor[scale], lp, gabor[scale]);
        mt = mt * mult;
    }
    const double angle_const = static_cast<double>(M_PI) / static_cast<double>(norient);
    for (int ori = 0; ori < norient; ori++)
    {
        double angl = (double)ori * angle_const;
        //Now we calculate the angular component that controls the orientation selectivity of the filter.
        for (int i = 0; i < dft_M; i++)
        {
            auto angular_row = angular.ptr<double>(i);
            for (int j = 0; j < dft_N; j++)
            {
                double m = atan2(-((double)j / (double)dft_N - 0.5), (double)i / (double)dft_M - 0.5);
                double s = sin(m);
                double c = cos(m);
                m = s * cos(angl) - c * sin(angl);
                double n = c * cos(angl) + s * sin(angl);
                s = fabs(atan2(m, n));

                angular_row[j] = (cos(__min(s * (double)norient * 0.5, M_PI)) + 1.0) * 0.5;
            }
        }
        for (int scale = 0; scale < nscale; scale++)
        {
            multiply(gabor[scale], angular, matAr[0]); //Product of the two components.
            merge(matAr, 2, filter[scale][ori]);
        }//scale
    }//orientation
    //Filter ready
}

void PhaseCongruency::calc(InputArray _src, OutputArray _edges, OutputArray _corners)
{
    Mat src = _src.getMat();
    _edges.create(src.size(), src.type());
    _corners.create(src.size(), src.type());
    auto edges = _edges.getMat();
    auto corners = _corners.getMat();
    int width = src.size().width, height = src.size().height;

    Mat src64;
    src.convertTo(src64, MAT_TYPE_CNV, 1.0 / 255.0);

    int64 freq = int64(getTickFrequency() / 1000.0);
    int64 t = getTickCount();
    
    auto dft_M = getOptimalDFTSize(src.rows);
    auto dft_N = getOptimalDFTSize(src.cols);

    cv::Mat pc[norient];

    cv::Mat dft_A;
    cv::Mat complex[2];
    cv::Mat eo[nscale][norient];
    cv::Mat sumAn;
    cv::Mat sumRe;
    cv::Mat sumIm;
    cv::Mat maxAn;
    cv::Mat xEnergy;
    cv::Mat tmp1;
    cv::Mat tmp2;
    cv::Mat tmp3;

    cv::Mat minMoment, maxMoment;

    cv::Mat energy = Mat::zeros(src.size(), MAT_TYPE);
    cv::Mat covx2 = Mat::zeros(src.size(), MAT_TYPE);
    cv::Mat covy2 = Mat::zeros(src.size(), MAT_TYPE);
    cv::Mat covxy = Mat::zeros(src.size(), MAT_TYPE);
    
    //expand input image to optimal size
    Mat padded;
    copyMakeBorder(src64, padded, 0, dft_M - src.rows, 0, dft_N - src.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = { Mat_<double>(padded), Mat::zeros(padded.size(), MAT_TYPE_CNV) };
    merge(planes, 2, dft_A);         // Add to the expanded another plane with zeros

    dft(dft_A, dft_A);            // this way the result may fit in the source matrix

    shiftDFT(dft_A, dft_A);
    auto dt = (getTickCount() - t) / freq;
    //std::cout << "DFT: " << dt << std::endl;
    t = getTickCount();

    const double angle_const = M_PI / (double)norient;

    for (int o = 0; o < norient; o++)
    {
        double noise = 0;
        for (int scale = 0; scale < nscale; scale++)
        {
            mulSpectrums(dft_A, filter[scale][o], tmp3, 0); // Convolution

            dft(tmp3, tmp3, DFT_INVERSE);
            tmp3(Rect(0, 0, width, height)).copyTo(eo[scale][o]);

            split(eo[scale][o], complex);

            magnitude(complex[0], complex[1], tmp1);

            if (scale == 0)
            {
                //here to do noise threshold calculation
                auto tau = mean(tmp1);
                tau.val[0] = tau.val[0] / sqrt(log(4.0));
                auto mt = 1.0 * pow(mult, nscale);
                auto totalTau = tau.val[0] * (1.0 - 1.0 / mt) / (1.0 - 1.0 / mult);
                auto m = totalTau * sqrt(M_PI / 2.0);
                auto n = totalTau * sqrt((4 - M_PI) / 2.0);
                noise = m + k * n;
                //xnoise = 0;
                //complex[0] -= xnoise;
                //max(complex[0], 0.0, complex[0]);

                tmp1.copyTo(maxAn);
                tmp1.copyTo(sumAn);
                complex[0].copyTo(sumRe);
                complex[1].copyTo(sumIm);
            }
            else
            {
                //complex[0] -= xnoise;
                //max(complex[0], 0.0, complex[0]);
                add(sumAn, tmp1, sumAn);
                max(tmp1, maxAn, maxAn);
                add(sumRe, complex[0], sumRe);
                add(sumIm, complex[1], sumIm);
            }
        } // next scale
        
        magnitude(sumRe, sumIm, xEnergy);
        xEnergy += epsilon;
        divide(sumIm, xEnergy, sumIm);
        divide(sumRe, xEnergy, sumRe);
        energy.setTo(0);
        for (int scale = 0; scale < nscale; scale++)
        {
            split(eo[scale][o], complex);

            multiply(complex[0], sumIm, tmp1);
            multiply(complex[1], sumRe, tmp2);

            absdiff(tmp1, tmp2, tmp2);
            subtract(energy, tmp2, energy);

            multiply(complex[0], sumRe, complex[0]);
            add(energy, complex[0], energy);
            multiply(complex[1], sumIm, complex[1]);
            add(energy, complex[1], energy);
            /*if (o == 0 && scale == 2)
            {
                energy -= noise / norient;
                max(energy, 0.0, energy);
                normalize(energy, tmp1, 0, 1, NORM_MINMAX);

                imshow("energy", tmp1);

            }*/
        } //next scale

        energy -= Scalar::all(noise); // -noise
        max(energy, 0.0, energy);
        maxAn += epsilon;

        divide(sumAn, maxAn, tmp1, double(1.0 / nscale));

        /*if (o == 3)
        {
        normalize(sumAn, tmp2, 0, 1, NORM_MINMAX);
        imshow("sumAn", tmp2);
        minMaxLoc(maxAn, &ml, &ML);
        std::cout << "maxAn min " << ml << " max " << ML << std::endl;
        normalize(maxAn, tmp2, 0, 1, NORM_MINMAX);
        imshow("maxAn", tmp2);
        minMaxLoc(tmp1, &ml, &ML);
        std::cout << "tmp1 min " << ml << " max " << ML << std::endl;
        max(tmp1, 0.0, tmp1);
        normalize(tmp1, tmp2, 0, 1, NORM_MINMAX);

        imshow("divider", tmp2);
        }*/

        tmp1 = tmp1 * double(-1.0);
        tmp1 += cutOff;   //cvScale(tmp1, tmp1, -g / ((double) nscale), cutOff * g);
        tmp1 = tmp1 * g;
        exp(tmp1, tmp1);
        tmp1 += 1.0; // 1 / weight

        //PC
        multiply(tmp1, sumAn, tmp1);
        divide(energy, tmp1, pc[o]);
        
        double angl = (double)o * angle_const;
        double sina = (double)sin(angl);
        double cosa = (double)cos(angl);

        //Build up covariance data for every point
        tmp1 = pc[o] * cosa;
        tmp2 = pc[o] * sina;
        multiply(tmp1, tmp2, complex[0]);
        add(covxy, complex[0], covxy);
        pow(tmp1, 2, tmp1);
        add(covx2, tmp1, covx2);
        pow(tmp2, 2, tmp2);
        add(covy2, tmp2, covy2);

        dt = (getTickCount() - t) / freq;
        //printf("Calc orient: %d ms\n", dt);
        t = getTickCount();
        //if (o == 0)
        //{
        //	imshow("orinetation", pc[o]);
        //}
    } // next orientation

    //Edges calculations
    covx2 *= 2.0 / static_cast<double>(norient);
    covy2 *= 2.0 / static_cast<double>(norient);
    covxy *= 4.0 / static_cast<double>(norient);

    subtract(covx2, covy2, tmp1);
      
    //tmp2 += Scalar::all(epsilon);
    magnitude(tmp1, covxy, tmp2); // denom;

    add(covy2, covx2, tmp1);
    subtract(tmp1, tmp2, minMoment);//m = (covy2 + covx2 - denom) / 2;          % ... and minimum moment
    add(tmp1, tmp2, maxMoment); //M = (covy2+covx2 + denom)/2;          % Maximum moment

    maxMoment.convertTo(edges, CV_8U, 255);
    minMoment.convertTo(corners, CV_8U, 255);
}


