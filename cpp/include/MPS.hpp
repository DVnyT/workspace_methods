#pragma once

#include "Index.hpp"
#include "SiteIndex.hpp"
#include "SiteTensor.hpp"
#include "Tensor.hpp"

class MPS
{
      private:
        // Main Objects =>
        int64_t m_physExtent;
        std::vector<SiteTensor> m_sites;

        int64_t m_maxBondExtent{-1};   // Default is -1 for no truncation of bond dimensions.
	double m_truncationCutoff{-1}; 

        // Derived Objects =>
        int m_siteNumber;                   // Number of sites = m_sites.size()
        std::vector<int64_t> m_bondExtents; // Looks like {1, 2, 4, 8, 16, 32, 50, 50, ... ,50, 50, 32, 16, 8, 4, 2, 1}
                                            // for physical dimension = 2, max bond dimension = 50.
                                            // SiteTensors will have corresponding extents ({Left, Phys, Right}):
                                            // {1, 2, 2}, {2, 2, 4}, {4, 2, 8}, ... , {32, 2, 50}, {50, 2, 50} ...

        // Data on the Host (CPU) =>

        // Data on the Device (GPU) =>
	cutensornetDescriptor_t m_descNet;

        // Constructors =>
        MPS(int siteNumber, int64_t physExtent);   // Randomly chosen coefficients
        MPS(const std::vector<SiteTensor>& sites); // Randomly chosen coefficients
	

        // Destructors =>
        // Copy/Move =>

        // Operations =>
        void leftNormalizeAll(); 
        void rightNormalizeAll();
        void mixedNormalize(int singularSite);

        void trace();
        void innerProduct();
        void transpose(); // cuTT(?)
};
