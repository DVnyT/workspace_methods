#pragma once

#include "../include/Index.hpp"

// Globals =>
std::atomic<size_t> Index::s_globalID{0};

// Constructors =>
Index::Index()
: m_uniqueID(s_globalID.fetch_add(1, std::memory_order_relaxed))
{}	

Index::Index(const int& dim)
: m_dim(dim), m_uniqueID(s_globalID.fetch_add(1, std::memory_order_relaxed))
{}

Index::Index(const int& dim, const std::vector<std::string>& tags)
: m_dim(dim), m_tags(tags), m_uniqueID(s_globalID.fetch_add(1, std::memory_order_relaxed))
{}

Index::Index(const int& dim, const int& prime)
: m_dim(dim), m_prime(prime), m_uniqueID(s_globalID.fetch_add(1, std::memory_order_relaxed))
{}

Index::Index(const int& dim, const std::vector<std::string>& tags, const int& prime)
: m_dim(dim), m_tags(tags), m_prime(prime), m_uniqueID(s_globalID.fetch_add(1, std::memory_order_relaxed))
{}

// Getters =>
int Index::getDim() const {return this->m_dim;}
const std::vector<std::string>& Index::getTags() const {return this->m_tags;}
int Index::getPrime() const {return this->m_prime;}
size_t Index::getUniqueID() const {return this->m_uniqueID;}
