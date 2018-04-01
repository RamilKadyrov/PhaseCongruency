

//#include "opencv2/core/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <iostream>

#include "phase.h"

#define _USE_MATH_DEFINES
#include <math.h>

//#include "options.h"

#define CV_MINMAX       32

// Rearrange the quadrants of Fourier image so that the origin is at
// the image center
// src & dst arrays of equal size & type
//void cvShiftDFT(CvArr * src_arr, CvArr * dst_arr)
void shiftDFT(cv::InputArray _src, cv::OutputArray _dst)
{
    cv::Mat tmp;
    cv::Mat q1stub, q2stub;
    cv::Mat q3stub, q4stub;
    cv::Mat d1stub, d2stub;
    cv::Mat d3stub, d4stub;
    cv::Mat s1, s2, s3, s4;
    cv::Mat d1, d2, d3, d4;
    cv::Mat src = _src.getMat();

    _dst.create(src.size(), src.type());

    auto dst = _dst.getMat();

    cv::Size size = src.size();
    cv::Size dst_size = dst.size();
    int cx, cy;

    /*if (dst_size.width != size.width ||
        dst_size.height != size.height){
        cvError(CV_StsUnmatchedSizes, "cvShiftDFT", "Source and Destination arrays must have equal sizes", __FILE__, __LINE__);
        }*/
    cx = size.width / 2;
    cy = size.height / 2; // image center

    //tmp.create(cx, cy, src.type());

    s1 = src(cv::Rect(0, 0, cx, cy));
    s2 = src(cv::Rect(cx, 0, cx, cy));
    s3 = src(cv::Rect(cx, cy, cx, cy));
    s4 = src(cv::Rect(0, cy, cx, cy));

    d1 = dst(cv::Rect(0, 0, cx, cy));
    d2 = dst(cv::Rect(cx, 0, cx, cy));
    d3 = dst(cv::Rect(cx, cy, cx, cy));
    d4 = dst(cv::Rect(0, cy, cx, cy));

    s3.copyTo(tmp);
    s1.copyTo(d3);
    tmp.copyTo(d1);
    s4.copyTo(tmp);
    s2.copyTo(d4);
    tmp.copyTo(d2);
}

// src & dst arrays of equal size & type
//void cvPhaseCongruency(const void* srcarr, void* dstarr)

//#define MAT_TYPE CV_64FC1
#define MAT_TYPE CV_64FC1
#define MAT_TYPE_CNV CV_64F
#define REAL  double

void PhaseCongruency::line(cv::OutputArray _dst, float angle)
{
    // Bresenham's line algorithm
    if (abs(angle) >= 2 * M_PI) throw std::exception("Angle bigger than 2*pi");

    int x1, y1, x0, y0;
    _dst.create(int(suppres_size), int(suppres_size), CV_8U);
    auto dst = _dst.getMat();
    dst.setTo(0);

    int width = dst.size().width;
    int height = dst.size().height;
    double cx = (double)width / 2.0;
    double cy = (double)height / 2.0;


    if (abs(angle) < 0.001 || abs(angle - 2 * M_PI) < 0.001)
    {
        for (int i = 0; i < width; ++i)
        {
            dst.at<unsigned char>(static_cast<int>(round(cy)), i) = 1;
        }
        return;
    }
    if (abs(M_PI * 0.5 - angle) < 0.001 || abs(M_PI *  1.5 - angle) < 0.001)
    {
        for (int i = 0; i < height; ++i)
        {
            dst.at<unsigned char>(i, static_cast<int>(round(cx))) = 1;
        }
        return;
    }
    --width;
    --height;
    double t = tan(angle);
    double x = 0;
    double y = cy + (x - cx) * t;
    if (y < 0)
    {
        y = 0;
        x = cx + (y - cy) * t;
    }
    else
    {
        if (y > height)
        {
            y = height;
            x = cx + (y - cy) * t;
        }
    }

    x0 = (int)x;
    y0 = (int)y;

    x = width;
    y = cy + (x - cx) * t;
    if (y > height)
    {
        y = height;
        x = cx + (y - cy) * t;
    }
    else
    {
        if (y < 0)
        {
            y = 0;
            x = cx + (y - cy) * t;
        }
    }

    x1 = (int)x;
    y1 = (int)y;

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2, e2;

    for (;;) {
        if (x0 >= 0 && x0 <= width && y0 >= 0 && y0 <= height)
            dst.at<unsigned char>(y0, x0) = 1;
        if (x0 == x1 && y0 == y1) break;
        e2 = err;
        if (e2 > -dx) { err -= dy; x0 += sx; }
        if (e2 < dy) { err += dx; y0 += sy; }
    }
}

PhaseCongruency::PhaseCongruency(cv::Size _img_size)
{

    auto dft_M = cv::getOptimalDFTSize(_img_size.height);
    auto dft_N = cv::getOptimalDFTSize(_img_size.width);

    cv::Mat radius;
    cv::Mat im;
    cv::Mat lp;
    cv::Mat angular;
    cv::Mat gabor[nscale];
    cv::Mat filter_ar[2];
    cv::Mat timg, tmp, troi;
    cv::Rect roi;
    //cv::Mat tmask[2];
    cv::Mat		tfilter[nscale][norient];
    // Prepare filter-------------------------
    //Matrix values contain *normalised* radius 
    // values ranging from 0 at the centre to 
    // 0.5 at the boundary.

    radius.create(dft_M, dft_N, MAT_TYPE);
    lp.create(dft_M, dft_N, MAT_TYPE);
    angular.create(dft_M, dft_N, MAT_TYPE);
    im.create(dft_M, dft_N, MAT_TYPE);
    filter_ar[0].create(dft_M, dft_N, MAT_TYPE);
    filter_ar[1].create(dft_M, dft_N, MAT_TYPE);

    im.setTo(0);
    radius.setTo(0);
    int r;
    int dft_M_2 = dft_M / 2;
    int dft_N_2 = dft_N / 2;
    if (dft_M > dft_N) r = dft_N_2;
    else r = dft_M_2;
    REAL dr = 1.0f / REAL(r);

    for (int row = dft_M_2 - r; row < dft_M_2 + r; row++)
    {
        auto radius_row = radius.ptr<REAL>(row);
        for (int col = dft_N_2 - r; col < dft_N_2 + r; col++)
        {
            REAL m = (REAL)(row - dft_M_2) * dr;
            REAL n = (REAL)(col - dft_N_2) * dr;
            REAL s = m * m + n * n;
            radius_row[col] = s;
        }
    }
    cv::sqrt(radius, radius);
    lp = radius * 2.5; // (1.0 / 0.4);
    cv::pow(lp, 20, lp);
    lp += cv::Scalar::all(1.0);  //cvAddS(lp, cvScalarAll(1.0), lp);
    //cvShowImage("magnitude", lp);
    radius.at<REAL>(dft_M_2, dft_N_2) = 1.0f;//  cvSetReal2D(radius, dft_M / 2, dft_N / 2, 1.0);
    // The following implements the log-gabor transfer function.
    REAL mt = 1.0f;
    for (int scale = 0; scale < nscale; scale++)
    {

        REAL wavelength = minwavelength * mt;

        gabor[scale].create(dft_M, dft_N, MAT_TYPE);

        gabor[scale] = radius * wavelength; //cvScale(radius, gabor[i], wavelength, 0);
        cv::log(gabor[scale], gabor[scale]);
        cv::pow(gabor[scale], 2, gabor[scale]);
        gabor[scale] *= sigma; //cvScale(gabor[i], gabor[i], sigma);
        cv::exp(gabor[scale], gabor[scale]);
        gabor[scale].at<REAL>(dft_M_2, dft_N_2) = 0.0f; // cvSetReal2D(gabor[i], dft_M / 2, dft_N / 2, 0.0);

        cv::divide(gabor[scale], lp, gabor[scale]);
        mt = mt * mult;

    }
    const REAL angle_const = static_cast<REAL>(M_PI) / static_cast<REAL>(norient);
    for (int ori = 0; ori < norient; ori++)
    {
        REAL angl = (REAL)ori * angle_const;

        //Now we calculate the angular component that controls the orientation selectivity of the filter.
        for (int i = 0; i < dft_M; i++)
        {
            auto angular_row = angular.ptr<REAL>(i);
            for (int j = 0; j < dft_N; j++)
            {
                REAL m = atan2(-((REAL)j / (REAL)dft_N - 0.5), (REAL)i / (REAL)dft_M - 0.5);
                REAL s = sin(m);
                REAL c = cos(m);
                m = s * cos(angl) - c * sin(angl);
                REAL n = c * cos(angl) + s * sin(angl);
                s = fabs(atan2(m, n));

                angular_row[j] = (cos(__min(s * (REAL)norient * 0.5, M_PI)) + 1.0) * 0.5;
            }
        }

        filter_ar[1].setTo(0);
        roi.x = dft_N_2;
        roi.y = 0;
        roi.height = dft_M;
        roi.width = dft_N_2;
        REAL rd;
        if (dft_M > dft_N) rd = dft_M; else rd = dft_N;

        for (int scale = 0; scale < nscale; scale++)
        {
            cv::multiply(gabor[scale], angular, radius); //Product of the two components.

            radius.copyTo(filter_ar[0]);
            filter_ar[1].setTo(0);
            cv::merge(filter_ar, 2, tfilter[scale][ori]);//cvMerge(radius, im, NULL, NULL, filter);// <----- ???? real and imaginary oro only real ?

            cv::dft(tfilter[scale][ori], timg, cv::DFT_INVERSE);
            
            cv::dft(timg, filter[scale][ori]);
        }//scale
    }//orientation
    //Filter ready ---------------------------------------------------------------------------------------------------

    for (int i = 0; i < 180; ++i)
    {
        REAL t_angl = M_PI * (REAL)i / 180.0;
        xoff(i) = thin_radius * cos(t_angl);
        yoff(i) = thin_radius * sin(t_angl);
        hfrac(i) = xoff(i) - floor(xoff(i));
        vfrac(i) = yoff(i) - floor(yoff(i));
    }
}

void PhaseCongruency::suppres(cv::OutputArray _dst)
{

    int height = pc[0].size().height;
    int width = pc[0].size().width;
    _dst.create(pc[0].size(), MAT_TYPE);
    auto dst = _dst.getMat();
    for (int i = 0; i < width; ++i)
    {
        for (int j = 0; j < height; ++j)
        {
            for (int o = 0; o < norient - 2; o += 2)
            {
                //int index;
                auto a0 = pc[o].at<REAL>(j, i);
                auto a1 = pc[o + 1].at<REAL>(j, i);
                auto a2 = pc[o + 2].at<REAL>(j, i);
                if (a0 > 0.5 && a0 > a1 && a0 > a2)
                {
                    dst.at<REAL>(j, i) = 1;
                }
            }
        }
    }
}

void PhaseCongruency::calc(cv::InputArray _src, cv::OutputArray _edges, cv::OutputArray _corners)
{
    cv::Mat src = _src.getMat();
    _edges.create(src.size(), src.type());
    _corners.create(src.size(), src.type());
    auto edges = _edges.getMat();
    auto corners = _corners.getMat();
    int width = src.size().width, height = src.size().height;

    cv::Mat src64;
    src.convertTo(src64, MAT_TYPE_CNV, 1.0 / 255.0);

    int64 dt = 0;
    int64 freq = int64(cv::getTickFrequency() / 1000.0);
    int64 t = cv::getTickCount();

    REAL mt;
    REAL noise;
    // REAL xnoise;

    int dft_M, dft_N;

    REAL m, n;
    //REAL M;
    //double ml;
    //double ML;
    cv::Point pMin, pMax;

    cv::Scalar tau;
    REAL totalTau;

    //if (CV_MAT_TYPE(src->type) != CV_8UC1 ||
    //	CV_MAT_TYPE(dst->type) != CV_8UC1)
    //	//CV_Error( CV_StsUnsupportedFormat, "" )
    //	;

    //if (!CV_ARE_SIZES_EQ(src, dst))
    //	CV_Error(CV_StsUnmatchedSizes, "");

    dft_M = cv::getOptimalDFTSize(src.rows);
    dft_N = cv::getOptimalDFTSize(src.cols);

    sumIm.create(src.size(), MAT_TYPE);
    sumRe.create(src.size(), MAT_TYPE);
    sumAn.create(src.size(), MAT_TYPE);
    maxAn.create(src.size(), MAT_TYPE);
    energy.create(src.size(), MAT_TYPE);
    tmp1.create(src.size(), MAT_TYPE);
    tmp2.create(src.size(), MAT_TYPE);
    covx2.create(src.size(), MAT_TYPE);
    covy2.create(src.size(), MAT_TYPE);
    covxy.create(src.size(), MAT_TYPE);

    covx2.setTo(0);
    covy2.setTo(0);
    covxy.setTo(0);


    //expand input image to optimal size
    cv::Mat padded;
    copyMakeBorder(src64, padded, 0, dft_M - src.rows, 0, dft_N - src.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    cv::Mat planes[] = { cv::Mat_<REAL>(padded), cv::Mat::zeros(padded.size(), MAT_TYPE_CNV) };
    merge(planes, 2, dft_A);         // Add to the expanded another plane with zeros

    cv::dft(dft_A, dft_A);            // this way the result may fit in the source matrix

    shiftDFT(dft_A, dft_A);
    dt = (cv::getTickCount() - t) / freq;
    //std::cout << "DFT: " << dt << std::endl;
    t = cv::getTickCount();

    const REAL angle_const = M_PI / (REAL)norient;

    for (int o = 0; o < norient; o++)
    {
        for (int scale = 0; scale < nscale; scale++)
        {
            cv::mulSpectrums(dft_A, filter[scale][o], tmp3, 0); // Convolution

            cv::dft(tmp3, tmp3, cv::DFT_INVERSE);
            tmp = tmp3(cv::Rect(0, 0, width, height));

            tmp.copyTo(eo[scale][o]);

            cv::split(eo[scale][o], complex);

            cv::pow(complex[0], 2.0, tmp1);
            cv::pow(complex[1], 2.0, tmp2);
            cv::add(tmp1, tmp2, tmp1);
            cv::sqrt(tmp1, tmp1);

            if (scale == 0)
            {
                //here to do noise threshold calculation
                tau = cv::mean(tmp1);
                tau.val[0] = tau.val[0] / sqrt(log(4.0));
                mt = 1.0 * pow(mult, nscale);
                totalTau = tau.val[0] * (1.0 - 1.0 / mt) / (1.0 - 1.0 / mult);
                m = totalTau * sqrt(M_PI / 2.0);
                n = totalTau * sqrt((4 - M_PI) / 2.0);
                noise = m + k * n;
                //xnoise = 0;
                //complex[0] -= xnoise;
                //cv::max(complex[0], 0.0, complex[0]);
                tmp1.copyTo(maxAn);
                tmp1.copyTo(sumAn);
                complex[0].copyTo(sumRe);
                complex[1].copyTo(sumIm);
            }
            else
            {
                //complex[0] -= xnoise;
                //cv::max(complex[0], 0.0, complex[0]);
                cv::add(sumAn, tmp1, sumAn);
                cv::max(tmp1, maxAn, maxAn);
                cv::add(sumRe, complex[0], sumRe);
                cv::add(sumIm, complex[1], sumIm);
            }
        } // next scale
        cv::pow(sumRe, 2.0, tmp1);
        cv::pow(sumIm, 2.0, tmp2);
        cv::add(tmp1, tmp2, xEnergy);
        cv::sqrt(xEnergy, xEnergy);
        xEnergy += epsilon;
        cv::divide(sumIm, xEnergy, sumIm);
        cv::divide(sumRe, xEnergy, sumRe);
        energy.setTo(0);
        for (int scale = 0; scale < nscale; scale++)
        {
            cv::split(eo[scale][o], complex);

            cv::multiply(complex[0], sumIm, tmp1);
            cv::multiply(complex[1], sumRe, tmp2);

            cv::absdiff(tmp1, tmp2, tmp2);
            cv::subtract(energy, tmp2, energy);

            cv::multiply(complex[0], sumRe, complex[0]);
            cv::add(energy, complex[0], energy);
            cv::multiply(complex[1], sumIm, complex[1]);
            cv::add(energy, complex[1], energy);
            /*if (o == 0 && scale == 2)
            {
                energy -= noise / norient;
                cv::max(energy, 0.0, energy);
                cv::normalize(energy, tmp1, 0, 1, cv::NORM_MINMAX);

                cv::imshow("energy", tmp1);

            }*/
        } //next scale

        energy -= cv::Scalar::all(noise); // -noise
        cv::max(energy, 0.0, energy);
        maxAn += epsilon;

        cv::divide(sumAn, maxAn, tmp1, REAL(1.0 / nscale));

        /*if (o == 3)
        {
        cv::normalize(sumAn, tmp2, 0, 1, cv::NORM_MINMAX);
        cv::imshow("sumAn", tmp2);
        cv::minMaxLoc(maxAn, &ml, &ML);
        std::cout << "maxAn min " << ml << " max " << ML << std::endl;
        cv::normalize(maxAn, tmp2, 0, 1, cv::NORM_MINMAX);
        cv::imshow("maxAn", tmp2);
        cv::minMaxLoc(tmp1, &ml, &ML);
        std::cout << "tmp1 min " << ml << " max " << ML << std::endl;
        cv::max(tmp1, 0.0, tmp1);
        cv::normalize(tmp1, tmp2, 0, 1, cv::NORM_MINMAX);

        cv::imshow("divider", tmp2);
        }*/

        tmp1 = tmp1 * double(-1.0);
        tmp1 += cutOff;   //cvScale(tmp1, tmp1, -g / ((REAL) nscale), cutOff * g);
        tmp1 = tmp1 * g;
        cv::exp(tmp1, tmp1);
        tmp1 += 1.0; // 1 / weight

        //PC
        cv::multiply(tmp1, sumAn, tmp1);
        cv::divide(energy, tmp1, pc[o]);
        
        REAL angl = (REAL)o * angle_const;
        REAL sina = (REAL)sin(angl);
        REAL cosa = (REAL)cos(angl);

        //Build up covariance data for every point
        tmp1 = pc[o] * cosa;
        tmp2 = pc[o] * sina;
        cv::multiply(tmp1, tmp2, complex[0]);
        cv::add(covxy, complex[0], covxy);
        cv::pow(tmp1, 2, tmp1);
        cv::add(covx2, tmp1, covx2);
        cv::pow(tmp2, 2, tmp2);
        cv::add(covy2, tmp2, covy2);

        dt = (cv::getTickCount() - t) / freq;
        //printf("Calc orient: %d ms\n", dt);
        t = cv::getTickCount();
        //if (o == 0)
        //{
        //	cv::imshow("orinetation", pc[o]);
        //}
    } // next orientation

    //Edges calculations

    covx2 *= 2.0 / (REAL)norient; //cvScale(covx2, covx2, 2.0 / (REAL) norient);
    covy2 *= 2.0 / (REAL)norient;//cvScale(covy2, covy2, 2.0 / (REAL) norient);
    covxy *= 4.0 / (REAL)norient;//cvScale(covxy, covxy, 4.0 / (REAL) norient);

    cv::subtract(covx2, covy2, tmp1);
    cv::pow(tmp1, 2, tmp1);
    cv::pow(covxy, 2, tmp2);
    cv::add(tmp2, tmp1, tmp2);
    tmp2 += cv::Scalar::all(epsilon);
    cv::sqrt(tmp2, tmp2); //denom

    cv::add(covy2, covx2, tmp1);

    cv::subtract(tmp1, tmp2, minMoment);//m = (covy2 + covx2 - denom) / 2;          % ... and minimum moment

    cv::add(tmp1, tmp2, maxMoment); //M = (covy2+covx2 + denom)/2;          % Maximum moment

    maxMoment.convertTo(edges, CV_8U, 255);

    minMoment.convertTo(corners, CV_8U, 255);
}


