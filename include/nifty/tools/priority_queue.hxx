#pragma once

#include <queue>

namespace nifty {
namespace tools{



/** \brief Heap-based priority queue.
    Implementation taken from the vigra library
    <b>\#include</b> \<nifty/tools/priority_queue.hxx\><br>
    Namespace: nifty::tools
*/
template <class ValueType,
          class PriorityType,
          bool Ascending = false>  // std::priority_queue is descending
class PriorityQueue
{
    typedef std::pair<ValueType, PriorityType> ElementType;
    
    struct Compare
    {
        typename IfBool<Ascending, std::greater<PriorityType>, 
                                   std::less<PriorityType> >::type cmp;
        
        bool operator()(ElementType const & l, ElementType const & r) const
        {
            return cmp(l.second, r.second);
        }
    };
    
    typedef std::priority_queue<ElementType, std::vector<ElementType>, Compare> Heap;
    
    Heap heap_;
    
  public:
  
    typedef ValueType value_type;
    typedef ValueType & reference;
    typedef ValueType const & const_reference;
    typedef typename Heap::size_type size_type;
    typedef PriorityType priority_type;
    
        /** \brief Create empty priority queue.
        */
    PriorityQueue()
    : heap_()
    {}
    
        /** \brief Number of elements in this queue.
        */
    size_type size() const
    {
        return heap_.size();
    }
    
        /** \brief Queue contains no elements.
             Equivalent to <tt>size() == 0</tt>.
        */
    bool empty() const
    {
        return size() == 0;
    }
    
        /** \brief Maximum index (i.e. priority) allowed in this queue.
             Equivalent to <tt>bucket_count - 1</tt>.
        */
    priority_type maxIndex() const
    {
        return NumericTraits<priority_type>::max();
    }
    
        /** \brief Priority of the current top element.
        */
    priority_type topPriority() const
    {
        return heap_.top().second;
    }
    
        /** \brief The current top element.
        */
    const_reference top() const
    {
        
        return heap_.top().first;
    }

        /** \brief Remove the current top element.
        */
    void pop()
    {
        heap_.pop();
    }
    
        /** \brief Insert new element \arg v with given \arg priority.
        */
    void push(value_type const & v, priority_type priority)
    {
        heap_.push(ElementType(v, priority));
    }
};


} // namespace nifty::tools
} // namespace nifty

