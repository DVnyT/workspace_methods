#include "catch_amalgamated.hpp"

#include "../include/Index.hpp"
#include <string>

TEST_CASE("Testing Constructors")
{
	SECTION("Default")
	{
		Index a();
		REQUIRE(a.getDim() == 1);
		REQUIRE(a.getPrime() == 0);
		REQUIRE(a.getCtorState() == 1);
		REQUIRE(a.getTags()[0][0] == '\0');
		REQUIRE(a.getTags()[3][7] == '\0');
		REQUIRE(a.getMode() != 0);
	}
	SECTION("2 default-constructed indices must not contract")
	{
		Index b(3, 2);
		Index c(3, 2);
		REQUIRE(a.getMode() != b.getMode());	
	}
	SECTION("isValidCollection m_tags Constructor")
	{
		std::vector<std::string> ver1 = {"tag1", "tag2", "7letter", "8letters"};
		std::array<std::string> ver2 =  {"tag1", "tag2", "7letter", "8letters"};
		std::string ver3[] =  {"tag1", "tag2", "7letter", "8letters"};
		std::string ver4[5] =  {"tag1", "tag2", "7letter", "8letters", "outofbounds!"};
		const char* ver5[M] = {"tag1", "tag2", "7letter", "8letters", "outofbounds!"};
		
		REQUIRE(isValidCollection<decltype(ver1)>);
		REQUIRE(isValidCollection<decltype(ver2)>);
		REQUIRE(isValidCollection<decltype(ver3)>);
		REQUIRE(isValidCollection<decltype(ver4)>);
		REQUIRE(isValidCollection<decltype(ver5)>);
		
		Index d(3, ver1);
		REQUIRE(d.getTags()[0][0] == 't');
		REQUIRE(d.getTags().size() == 4);
		REQUIRE(d.getTags()[3][7] == 's');
                
		Index e(3, ver2);
		REQUIRE(d.getTags()[0][0] == 't');
                REQUIRE(d.getTags().size() == 4);
                REQUIRE(d.getTags()[3][7] == 's');

		Index f(3, ver3);
                REQUIRE(e.getTags()[0][0] == 't');
                REQUIRE(e.getTags().size() == 4);
                REQUIRE(e.getTags()[3][7] == 's');

		Index g(3, ver4);
                REQUIRE(f.getTags()[0][0] == 't');
                REQUIRE(f.getTags().size() == 4);
                REQUIRE(f.getTags()[3][7] == 's');

		Index h(3, ver5);
                REQUIRE(g.getTags()[0][0] == 't');
                REQUIRE(g.getTags().size() == 4);
                REQUIRE(g.getTags()[3][7] == 's');
	}
}
