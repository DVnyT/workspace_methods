#include "catch_amalgamated.hpp"

#include "../include/Index.hpp"
#include <string>

TEST_CASE("Testing Constructors")
{

	Index a;
	SECTION("Default =>")
	{
		REQUIRE(a.getExtent() == 1);
		REQUIRE(a.getPrime() == 0);
		REQUIRE(a.getCtorState() == 1);
		REQUIRE(a.getMode() != 0);
	}

	Index b(3, 2);
	Index c(3, 2);
	SECTION("2 default-constructed indices must not contract =>")
	{
		REQUIRE(b.getMode() != c.getMode());	
	}
	
	Index d(3, 1);
	Index e(d);
	Index f = d;
	SECTION("Copy ctor and copy assignment is a deep copy =>")
	{
		REQUIRE(d.getExtent() == e.getExtent());
		REQUIRE(d.getMode() == e.getMode());
		REQUIRE(d == e);
		REQUIRE(d.getCtorState() != e.getCtorState());

		REQUIRE(d.getExtent() == f.getExtent());
		REQUIRE(d.getMode() == f.getMode());
		REQUIRE(d == f);
		REQUIRE(d.getCtorState() != f.getCtorState());
	}
	
	e.prime(2);
	f.setExtent(5);
	SECTION("Modified copies do not contract =>")
	{
		REQUIRE(d != e);
		REQUIRE(e.getCtorState() == 1);
		
		REQUIRE(d != e);
		REQUIRE(e.getCtorState() == 1);
	}

	Index g = f;
	Index h = f;	// f has extent = 5 and prime = 1 as of now
	g.setExtent(9);
	h.setExtent(9);
	SECTION("2 copies with the same modification do not contract =>")
	{
		REQUIRE(g != h);
	}
}
