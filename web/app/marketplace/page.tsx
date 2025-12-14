"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import ProductCard from "../components/card/ProductCard";
import { Slider } from "../../components/ui/slider";
import {
  Select,
  SelectContent,
  SelectItem,   
  SelectTrigger,
  SelectValue,
} from "../../components/ui/select";

interface Phone {
  id: string;
  uuid:string;
  name?: string;
  model: string;
  company?: string;
  ram?: string;
  storage?: string;
  pictures?: string[];
  condition_score?: number;
  description?: string;
  pta_status?: boolean;
  price?: number;
}
export default function Marketplace() {
  const [searchQuery, setSearchQuery] = React.useState("");
  const [filters, setFilters] = React.useState({
    company: "all",
    storage: "all",
    priceMin: 0,
    priceMax: 500000,
  });
  const [currentPage, setCurrentPage] = React.useState(1);
  const itemsPerPage = 12;

  const { data: allPhones = [], isLoading } = useQuery<Phone[]>({
    queryKey: ["marketplace-phones"],
    queryFn: async () => {
      const res = await fetch("/api/phones/list");
      console.log('Fetched phones:', res);
      return res.json();
    },
  });
  // Unique companies
  const companies = [...new Set(allPhones.map((p) => p.company).filter(Boolean))];

  // Filter Logic
  const filteredPhones = allPhones.filter((phone) => {
    const search = searchQuery.toLowerCase();

    const matchesSearch =
      !searchQuery ||
      phone.model?.toLowerCase().includes(search) ||
      phone.company?.toLowerCase().includes(search);

    const matchesCompany =
      filters.company === "all" || phone.company === filters.company;

    const matchesStorage =
      filters.storage === "all" || phone.storage === filters.storage;

    const matchesPrice =
      (phone.price ?? 0) >= filters.priceMin && (phone.price ?? 0) <= filters.priceMax;

    return matchesSearch && matchesCompany && matchesStorage && matchesPrice;
  });

  // Pagination
  const totalPages = Math.ceil(filteredPhones.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const paginatedPhones = filteredPhones.slice(
    startIndex,
    startIndex + itemsPerPage
  );

  return (
    <div className="min-h-screen py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1 className="text-4xl font-bold mb-8">Marketplace</h1>

        {/* Filters */}
        <div className="grid md:grid-cols-4 gap-6 mb-8">
          {/* Company Filter */}
          <div>
            <label className="block text-sm font-medium mb-2">Company</label>
            <Select
              value={filters.company}
              onValueChange={(v) => setFilters({ ...filters, company: v })}
            >
              <SelectTrigger className="bg-black border-gray-800">
                <SelectValue placeholder="Select company" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Companies</SelectItem>
                {companies.map((company) => (
                  <SelectItem key={company!} value={company!}>
                    {company}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* Storage Filter */}
          <div>
            <label className="block text-sm font-medium mb-2">Storage</label>
            <Select
              value={filters.storage}
              onValueChange={(v) => setFilters({ ...filters, storage: v })}
            >
              <SelectTrigger className="bg-black border-gray-800">
                <SelectValue placeholder="Select storage" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Storage</SelectItem>
                <SelectItem value="64GB">64GB</SelectItem>
                <SelectItem value="128GB">128GB</SelectItem>
                <SelectItem value="256GB">256GB</SelectItem>
                <SelectItem value="512GB">512GB</SelectItem>
                <SelectItem value="1TB">1TB</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Price Slider */}
          <div className="col-span-2">
            <label className="block text-sm font-medium mb-2">
              Price Range: Rs. {filters.priceMin.toLocaleString()} - Rs.{" "}
              {filters.priceMax.toLocaleString()}
            </label>
            <Slider
              min={0}
              max={500000}
              step={10000}
              value={[filters.priceMin, filters.priceMax]}
              onValueChange={([min, max]) =>
                setFilters({ ...filters, priceMin: min, priceMax: max })
              }
              className="mt-2"
            />
          </div>
        </div>

        {/* Products Grid */}
        {isLoading ? (
          <div className="text-center py-20">
            <div className="inline-block w-12 h-12 border-4 border-[#f7f435] border-t-transparent rounded-full animate-spin"></div>
          </div>
        ) : paginatedPhones.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6 mb-8">
            {paginatedPhones.map((phone) => (
              <ProductCard key={phone.id} phone={phone} />
            ))}
          </div>
        ) : (
          <div className="text-center py-20">
            <p className="text-xl text-gray-400">
              No phones found matching your criteria
            </p>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2">
            {Array.from({ length: totalPages }, (_, i) => i + 1).map(
              (page) => (
                <button
                  key={page}
                  onClick={() => setCurrentPage(page)}
                  className={`px-4 py-2 rounded-lg ${
                    currentPage === page
                      ? "yellow-btn"
                      : "glass-panel hover:bg-gray-800"
                  }`}
                >
                  {page}
                </button>
              )
            )}
          </div>
        )}
      </div>
    </div>
  );
}
