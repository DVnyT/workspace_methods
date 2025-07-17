#include "../include/MPS.hpp"
#include <vector>

MPS::MPS(int siteNumber, int64_t physExtent) : m_siteNumber(siteNumber), m_physExtent(physExtent)
{
        // TODO: Define default ctor for SiteTensor for resizing.
        // m_sites.resize(m_siteNumber);
        // m_bondExtents.resize(m_siteNumber)
        int tmp = 1;
        int i = 0;
        for (; i < m_siteNumber / 2; i++)
        {
                m_sites.emplace_back(tmp, m_physExtent, (tmp * m_physExtent));
                m_sites[i].setRand();
                m_bondExtents[i] = tmp;
                tmp *= physExtent;
        }
        for (; i < m_siteNumber; i++)
        {
                m_sites.emplace_back(tmp, m_physExtent, (tmp / m_physExtent));
                m_sites[i].setRand();
                m_bondExtents[i] = tmp;
                tmp /= physExtent;
        }
        // TODO: cutensorNetDescriptor_t init.
}

MPS::MPS(const std::vector<SiteTensor>& sites)
    : m_sites(sites), m_siteNumber(sites.size()), m_physExtent(sites[0].getPhysExtent())
{
        for (int i = 0; i < m_siteNumber; i++)
        {
                m_bondExtents[i] = m_sites[i].m_leftIndex.getExtent();
        }
}

const std::vector<int32_t> numModesIn(m_siteNumber, 3);
HANDLE_CUTENSORNET_ERROR(cutensornetCreateNetworkDescriptor(handle, m_siteNumber, numModesIn.data(),
                                                            /*stridesIn = */ NULL, modesIn, NULL, numModesOut,
                                                            extentOut.data(), NULL, modesOut.data(), CUDA_R_32F,
                                                            CUTENSORNET_COMPUTE_32F, &m_descNet));

MPS::~MPS() {}
